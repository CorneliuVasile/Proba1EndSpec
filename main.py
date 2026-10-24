"""Build a category mapping from the four input CSVs.

Run: python main.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'output'
# Audited proposals, not labels from an automated model. Keep the reasons here.
OVERRIDES = {
    'citr.frt': (1475, 'Berry product names dominate the category revenue in keyword audit'),
    'plant-based.drnks': (5179, 'Plant-based drink names dominate category revenue in keyword audit'),
    'sal:sml': (4782, 'Salad vegetable names dominate category revenue in keyword audit'),
}


def normalize(value):
    """Supplier ignores casing and whitespace (including embedded whitespace)."""
    return re.sub(r'\s+', '', str(value)).casefold()


def build_text(group):
    name = str(group['supplier_category'].iloc[0]).replace('_', ' ').replace('>', ' ')
    examples = group[['product_name', 'brand']].fillna('').drop_duplicates().head(35)
    return (name + ' ' + ' '.join((examples['product_name'] + ' ' + examples['brand']).tolist())).lower()


def generate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    products = pd.read_csv(DATA / 'supplier_products.csv', dtype={'supplier_sku': str})
    labels = pd.read_csv(DATA / 'labelled_products.csv', dtype={'supplier_sku': str})
    categories = pd.read_csv(DATA / 'our_categories.csv')
    valid_ids = set(categories.our_category_id)
    if products.supplier_sku.duplicated().any() or labels.supplier_sku.duplicated().any():
        raise ValueError('Duplicate product or label SKU')
    products['category_key'] = products.supplier_category.map(normalize)
    labelled = products[['supplier_sku','category_key']].merge(labels, on='supplier_sku', how='inner', validate='one_to_one')
    if len(labelled) != len(labels):
        raise ValueError('Unknown labelled SKU')
    if not labelled.our_category_id.isin(valid_ids).all():
        raise ValueError('Manual labels reference unknown category IDs')
    votes = (labelled.groupby(['category_key','our_category_id']).size().reset_index(name='votes')
             .sort_values(['category_key','votes','our_category_id'], ascending=[True,False,True]))
    winners = votes.drop_duplicates('category_key').set_index('category_key')['our_category_id'].sort_index()
    texts = products.groupby('category_key', sort=True).apply(build_text, include_groups=False)
    unknown = sorted(set(texts.index) - set(winners.index))
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=45000,
                                 sublinear_tf=True, strip_accents='unicode')
    # The original Step 3 mapping used vocabulary from all unlabeled texts as well:
    matrix = vectorizer.fit_transform(texts.values)
    idx = pd.Series(np.arange(len(texts)), index=texts.index)
    sims = cosine_similarity(matrix[idx.loc[unknown].to_numpy()], matrix[idx.loc[winners.index].to_numpy()])
    predictions = pd.Series(winners.to_numpy()[sims.argmax(axis=1)], index=unknown)
    key_ids = pd.concat([winners, predictions])
    for key, (category_id, reason) in OVERRIDES.items():
        if key not in key_ids.index or category_id not in valid_ids:
            raise ValueError(f'Invalid override: {key}, {category_id}')
        key_ids.loc[key] = category_id
    originals = products[['supplier_category','category_key']].drop_duplicates()
    result = originals.assign(our_category_id=originals.category_key.map(key_ids))
    result = result[['supplier_category', 'our_category_id']]
    validate(result, products, valid_ids)
    result.to_csv(OUTPUT / 'mapping.csv', index=False)
    return result, len(winners), len(unknown)


def validate(mapping, products, valid_ids):
    if mapping.columns.tolist() != ['supplier_category', 'our_category_id']:
        raise AssertionError('Unexpected output columns')
    if mapping.supplier_category.duplicated().any():
        raise AssertionError('Duplicate original category spelling')
    if set(mapping.supplier_category) != set(products.supplier_category):
        raise AssertionError('Missing/extra supplier category')
    if mapping.our_category_id.isna().any() or not mapping.our_category_id.isin(valid_ids).all():
        raise AssertionError('Invalid or null destination category ID')
    if mapping.assign(key=mapping.supplier_category.map(normalize)).groupby('key').our_category_id.nunique().max() != 1:
        raise AssertionError('Whitespace/case variants differ')


if __name__ == '__main__':
    result, known, unknown = generate()
    print(f'Created output/mapping.csv with {len(result)} original spellings.')
    print(f'Manually represented normalized categories: {known}; inferred: {unknown}.')
    print(f'Applied {len(OVERRIDES)} reviewed category overrides.')
