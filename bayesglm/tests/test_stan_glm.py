import numpy as np
import numpy.testing as nptest
from scipy.special import logit, expit

import pandas as pd
import unittest

from ..stan_glm import load_model_template, stan_glm
from ..priors import NormalPrior
from .. import family

BETA = np.array([15, 5])
NUM_ROWS = 2000
ITERATIONS = 200
PRIOR_BETA_MEAN1 = 10
PRIOR_BETA_MEAN2 = 15
PRIOR_BETA_MEAN3 = 2
PRIOR_BETA_VARIANCE = .00001


def make_matrix_data(num_rows, beta, noise_sd=1, binary=False):
    np.random.seed(seed=0)
    x = np.random.normal(size=(num_rows, len(beta)))
    y = np.dot(x, beta) + np.random.normal(size=num_rows, scale=noise_sd)
    if binary:
        y = np.random.binomial(n=1, p=expit(y))
    return x, y


def make_data_frame_data(num_rows, beta, noise_sd=1, binary=False):
    x, y = make_matrix_data(num_rows, beta, noise_sd=noise_sd, binary=binary)
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df['y'] = y
    return df


class ModelTests(unittest.TestCase):
    def test_load_model_template(self):
        self.assertTrue(type(load_model_template()) == str)

    def test_stan_glm_gaussian(self):
        # tests both data frame and matrix form
        x, y = make_matrix_data(num_rows=NUM_ROWS, beta=BETA)
        result_matrix = stan_glm(x, y, family=family.gaussian(), iterations=ITERATIONS, seed=0)
        beta_samples = result_matrix.extract()['beta']
        beta_means = beta_samples.mean(axis=0)
        nptest.assert_allclose(beta_means, np.array(BETA), atol=.5)

        # check that data frame result is same as matrix result
        df = make_data_frame_data(num_rows=NUM_ROWS, beta=BETA)
        result_data_frame = stan_glm("y ~ 0 + x1 + x2", df, family=family.gaussian(), iterations=ITERATIONS, seed=0)
        nptest.assert_allclose(result_matrix.extract(permuted=False), result_data_frame.extract(permuted=False))

    def test_stan_glm_gaussian_priors(self):
        iterations = 100
        x, y = make_matrix_data(num_rows=NUM_ROWS, beta=BETA)
        normal_prior = NormalPrior(PRIOR_BETA_MEAN1, PRIOR_BETA_VARIANCE)
        prior1 = (((i,), normal_prior) for i in [0, 1])
        prior2 = (((0, 1), normal_prior),)
        result1 = stan_glm(x, y, family=family.gaussian(), iterations=iterations, seed=0, priors=prior1)
        result2 = stan_glm(x, y, family=family.gaussian(), iterations=iterations, seed=0, priors=prior2)
        nptest.assert_allclose(result1.extract(permuted=False), result2.extract(permuted=False))
        beta_means = result1.extract()['beta'].mean(axis=0)
        nptest.assert_allclose(beta_means, np.array([PRIOR_BETA_MEAN1, PRIOR_BETA_MEAN1]), atol=.01)

    def test_stan_glm_logistic(self):
        df = make_data_frame_data(num_rows=NUM_ROWS, beta=BETA, binary=True)
        result = stan_glm("y ~ x1 + x2", df, family=family.bernoulli(), iterations=ITERATIONS, seed=0)
        beta_samples = result.extract()['beta']
        beta_means = beta_samples.mean(axis=0)
        true_betas = np.hstack([[0], BETA])
        nptest.assert_allclose(beta_means, true_betas, atol=1.5)  # "0" is true parameter for constant

    def test_stan_glm_gaussian_priors_formula(self):
        iterations = 100
        df = make_data_frame_data(num_rows=NUM_ROWS, beta=BETA)
        priors = {"x1": NormalPrior(PRIOR_BETA_MEAN1, PRIOR_BETA_VARIANCE),
                  "x2": NormalPrior(PRIOR_BETA_MEAN2, PRIOR_BETA_VARIANCE),
                  "x1:x2": NormalPrior(PRIOR_BETA_MEAN3, PRIOR_BETA_VARIANCE)}
        result = stan_glm("y ~ 0 + x1 + x2 + x1*x2", df, family=family.gaussian(), iterations=iterations, seed=0,
                          priors=priors)
        beta_means = result.extract()['beta'].mean(axis=0)
        nptest.assert_allclose(beta_means, np.array([PRIOR_BETA_MEAN1, PRIOR_BETA_MEAN2, PRIOR_BETA_MEAN3]), atol=.01)