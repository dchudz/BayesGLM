import unittest

from ..get_names import get_column_names, get_term_names

DATA = {'a': ['a1', 'a1', 'a2', 'a2', 'a3', 'a1', 'a2', 'a2'],
        'b': ['b1', 'b2', 'b1', 'b2', 'b1', 'b2', 'b1', 'b2'],
        'x1': [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799, -0.97727788, 0.95008842, -0.15135721],
        'x2': [-0.10321885, 0.4105985, 0.14404357, 1.45427351, 0.76103773, 0.12167502, 0.44386323, 0.33367433],
        'y': [1.49407907, -0.20515826, 0.3130677, -0.85409574, -2.55298982, 0.6536186, 0.8644362, -0.74216502]}


class GetNamesTests(unittest.TestCase):
    def test_get_term_names(self):
        term_names = get_term_names("y ~ x1 + x2 + x1*x2 + a", DATA)
        self.assertSetEqual(set(term_names), {'Intercept', 'a', 'x1', 'x2', 'x1:x2'})

    def test_get_column_names1(self):
        column_names = get_column_names("y ~ x1 + x2 + x1*x2 + a", DATA)
        self.assertSetEqual(set(column_names), {'Intercept', 'a[T.a2]', 'a[T.a3]', 'x1', 'x2', 'x1:x2'})

    def test_get_column_names2(self):
        column_names = get_column_names("y ~ x2 + a:x1", DATA)
        self.assertSetEqual(set(column_names), {'Intercept', 'x2', 'a[a1]:x1', 'a[a2]:x1', 'a[a3]:x1'})
