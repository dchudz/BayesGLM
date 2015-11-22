import numpy as np
from scipy.special import logit, expit
import pandas as pd
import unittest

from ..models import load_model_template, bayesglm
from .. import family

BETA = np.array([15, 5])
NUM_ROWS = 2000
ITERATIONS = 50

def make_matrix_data(num_rows, beta, noise_sd=1, binary=False):
    np.random.seed(seed=0)
    x = np.random.normal(size=(num_rows,len(beta)))
    y = np.dot(x, beta) + np.random.normal(size=num_rows, scale=noise_sd)
    if binary:
        y = np.random.binomial(n=1, p=expit(y))
    return x, y


def make_dataframe_data(num_rows, beta, noise_sd=1, binary=False):
    x, y = make_matrix_data(num_rows, beta, noise_sd=noise_sd, binary=binary)
    df = pd.DataFrame(x, columns = ["x1", "x2"])
    df['y'] = y
    return df


class Tests(unittest.TestCase):

    def test_load_model_template(self):
        self.assertTrue(type(load_model_template()) == str)

    def test_bayesglm_gaussian_matrix(self):
        x, y = make_matrix_data(num_rows=NUM_ROWS, beta=BETA)
        result = bayesglm(x, y, family=family.gaussian(), iterations=ITERATIONS, seed=0)
        beta_samples = result.extract()['beta']
        print beta_samples 

    def test_bayesglm_gaussian_dataframe(self):
        df = make_dataframe_data(num_rows=NUM_ROWS, beta=BETA)
        result = bayesglm("y ~ x1 + x2", df, family=family.gaussian())
        print result


    def test_bayesglm_logistic(self):
        df = make_dataframe_data(num_rows=NUM_ROWS, beta=BETA, binary=True)
        result = bayesglm("y ~ x1 + x2", df, family=family.bernoulli())
        print result


    def test_logistic(self):
        pass
