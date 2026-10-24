"""Category-level holdout, labels on test categories excluded from training."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from main import DATA, OUTPUT, normalize, build_text


def main():
    products = pd.read_csv(DATA / 'supplier_products.csv', dtype={'supplier_sku': str})
    labels = pd.read_csv(DATA / 'labelled_products.csv', dtype={'supplier_sku': str})
    sales = pd.read_csv(DATA / 'sales.csv', dtype={'supplier_sku': str})
    products['category_key'] = products.supplier_category.map(normalize)
    lab = products[['supplier_sku','category_key']].merge(labels,on='supplier_sku',validate='one_to_one')
    counts = (lab.groupby(['category_key','our_category_id']).size().reset_index(name='votes')
              .sort_values(['category_key','votes','our_category_id'],ascending=[True,False,True]))
    majority = counts.drop_duplicates('category_key').set_index('category_key').our_category_id.sort_index()
    texts = products.groupby('category_key',sort=True).apply(build_text,include_groups=False)
    revenue = (products[['supplier_sku','category_key']]
               .merge(sales[['supplier_sku','revenue']],on='supplier_sku',validate='one_to_one')
               .groupby('category_key').revenue.sum())
    train, test = train_test_split(majority.index.to_numpy(),test_size=.20,random_state=42)
    train, test = sorted(train), sorted(test)
    y_train, y_test = majority.loc[train].to_numpy(), majority.loc[test].to_numpy()
    common = pd.Series(y_train).value_counts().sort_values(ascending=False).index[0]
    tfidf = TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=45000,sublinear_tf=True,strip_accents='unicode')
    x_train = tfidf.fit_transform(texts.loc[train].values)
    x_test = tfidf.transform(texts.loc[test].values)
    nearest = y_train[cosine_similarity(x_test,x_train).argmax(axis=1)]
    ridge = RidgeClassifier(alpha=1.0).fit(x_train,y_train).predict(x_test)
    results = pd.DataFrame({'category_key': test,'reference_majority_id':y_test,
                            'baseline_id':np.full(len(test),common), 'tfidf_nearest_id':nearest,
                            'tfidf_ridge_id':ridge,'annual_revenue_eur':revenue.reindex(test).to_numpy()})
    for name, column in [('baseline','baseline_id'),('tfidf_nearest','tfidf_nearest_id'),('tfidf_ridge','tfidf_ridge_id')]:
        correct = results[column].to_numpy()==y_test
        results[name+'_correct'] = correct
        print(f'{name}: {correct.sum()}/{len(test)} = {correct.mean():.2%}; disagreement-associated revenue EUR {results.loc[~correct,"annual_revenue_eur"].sum():,.2f}')
    OUTPUT.mkdir(exist_ok=True)
    results.to_csv(OUTPUT/'evaluation.csv',index=False)
    print('Caveat: comparisons use category-majority labels, not independently verified ground truth.')

if __name__=='__main__': main()
