import unittest
from bayesglm.models import load_model_template


class Tests(unittest.TestCase):

    def test_load_model_template(self):
        self.assertTrue(type(load_model_template()) == str)
