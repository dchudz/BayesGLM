from bayesglm.stan_cache import stan_cache
from patsy import dmatrices
from multipledispatch import dispatch
import numpy as np
import os

from .family import Family


class NormalPrior:
    def __init__(self, mu, sigma):
        self.mu = mu;
        self.sigma = sigma

    def to_string(self):
        return "normal({0},{1})".format(self.mu, self.sigma)


def load_model_template():
    stan_model_path = os.path.join(os.path.dirname(__file__), "model.stan")
    with open(stan_model_path, "r") as stan_model_file:
        stan_model_template = stan_model_file.read()
    return stan_model_template


def parameter_priors_to_string(parameter_priors):
    # parameter_priors should be a list of tuples (r, prior) where r is a range and prior is a prior like NormalPrior
    #   This approach leads to different stan models depending on which parameters you choose to give which priors.
    #   We could probably avoid recompiling so much by switching to an approach where the stan code is always the same
    #   for any set of priors.
    model_string = ""
    for r, prior in parameter_priors:
        for i in r:
            model_string += "beta[{0}] ~ {1};\n".format(i+1, prior.to_string()) # stan is 1-indexed
    return model_string


def stan_code(family, beta_priors):
    return load_model_template().format(y_type=family.distribution.y_type,
                                        parameter_statement=family.distribution.parameter_statement,
                                        link_function=family.link,
                                        model_statement=family.distribution.model_statement,
                                        beta_priors=family.parameter_priors_to_string(beta_priors))


@dispatch(np.ndarray, np.ndarray, Family)
def bayesglm(x, y, family, iterations=100, priors=()):
    num_rows, num_predictors = x.shape
    model_code = stan_code(family, priors)
    print(model_code)
    fit = stan_cache(model_code=model_code,
                     data={"x": x, "N": num_rows, "K": num_predictors, "y": y},
                     iter=iterations,
                     chains=4)
    return fit


@dispatch(str, object, Family)
def bayesglm(formula, df, family, priors=None):
    if not priors:
        priors = {}
    y, x = dmatrices(formula, df)
    x_ = np.asarray(x)
    if family.distribution.y_type == "int":  # (necessary b/c patsy converts to float)
        y = y.astype("int")
    y_ = np.asarray(np.squeeze(y))

    def slice_to_range(s):
        return range(s.start, s.stop)
    beta_priors_list = [(slice_to_range(x.design_info.slice(key)), val) for key, val in priors.items()]
    return bayesglm(x_, y_, family, priors=beta_priors_list)