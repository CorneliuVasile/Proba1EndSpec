"""Revenue-weighted review queue for category mappings, not financial loss estimate."""
import pandas as pd
from main import DATA, OUTPUT, normalize


def main():
    products = pd.read_csv(DATA/'supplier_products.csv',dtype={'supplier_sku':str})
    sales = pd.read_csv(DATA/'sales.csv',dtype={'supplier_sku':str})
    labels = pd.read_csv(DATA/'labelled_products.csv',dtype={'supplier_sku':str})
    mapping = pd.read_csv(OUTPUT/'mapping.csv')
    products['category_key'] = products.supplier_category.map(normalize)
    mapping['category_key'] = mapping.supplier_category.map(normalize)
    lab = products[['supplier_sku','category_key']].merge(labels,on='supplier_sku',validate='one_to_one')
    stat = (lab.groupby('category_key').our_category_id.agg(n_labels='size',n_destinations='nunique'))
    revenue = (products[['supplier_sku','category_key']].merge(sales[['supplier_sku','revenue']],on='supplier_sku',validate='one_to_one').groupby('category_key').revenue.sum())
    result = mapping.drop_duplicates('category_key').set_index('category_key')[['supplier_category','our_category_id']]
    result = result.join(stat).join(revenue.rename('annual_revenue_eur'))
    result['n_labels']=result.n_labels.fillna(0).astype(int)
    result['n_destinations']=result.n_destinations.fillna(0).astype(int)
    result['review_reason']=result.apply(lambda x: 'no labelled examples' if x.n_labels==0 else ('contradictory labels' if x.n_destinations>1 else 'labelled'),axis=1)
    result['needs_review']=result.review_reason!='labelled'
    result=result.sort_values(['needs_review','annual_revenue_eur'],ascending=[False,False])
    OUTPUT.mkdir(exist_ok=True)
    result.reset_index().to_csv(OUTPUT/'review_queue.csv',index=False)
    print(f'Created output/review_queue.csv: {int(result.needs_review.sum())} categories flagged.')
    print('Revenue associated with review needs is NOT an estimate of lost sales.')

if __name__=='__main__': main()
