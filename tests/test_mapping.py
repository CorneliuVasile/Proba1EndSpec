import unittest
import pandas as pd
from main import DATA, OUTPUT, normalize, validate

class TestMapping(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize(' Ab  C '),normalize('a b c'))
    def test_output(self):
        products=pd.read_csv(DATA/'supplier_products.csv',dtype={'supplier_sku':str})
        categories=pd.read_csv(DATA/'our_categories.csv')
        mapping=pd.read_csv(OUTPUT/'mapping.csv')
        validate(mapping, products, set(categories.our_category_id))
        self.assertEqual(mapping.columns.tolist(),['supplier_category','our_category_id'])

if __name__=='__main__': unittest.main()
