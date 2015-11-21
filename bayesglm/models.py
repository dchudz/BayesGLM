from bayesglm.stan_cache import stan_cache
from collections import namedtuple
from patsy import dmatrices
from multipledispatch import dispatch
import numpy as np
import os


OutcomeDistribution = namedtuple('OutcomeDistribution', ['y_type', 'parameter_names', 'parameter_statement', 'model_statement'])


class OutcomeDistributions:
    gaussian = OutcomeDistribution(
        y_type = "real",
        parameter_names = ["sigma"],
        parameter_statement = "real sigma;",
        model_statement = "y ~ normal(mu, sigma);"
    )
    bernoulli_logit = OutcomeDistribution(
        y_type = "int",
        parameter_names = [],
        parameter_statement = "",
        model_statement = "y ~ bernoulli_logit(mu);"
    )
    bernoulli = OutcomeDistribution(
        y_type = "int",
        parameter_names = [],
        parameter_statement = "",
        model_statement = "y ~ bernoulli(mu);"
    )


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


class RegressionModel(object):
    def __init__(self, distribution, link):
        self.distribution = distribution
        self.link = link

    # parameter_priors should be a list of tuples (r, prior) where r is a range and prior is a prior like NormalPrior
    #   This approach leads to different stan models depending on which parameters you choose to give which priors.
    #   We could probably avoid recompiling so much by switching to an approach where the stan code is always the same
    #   for any set of priors.
    def parameter_priors_to_string(self, parameter_priors):
        model_string = ""
        for r, prior in parameter_priors:
            for i in r:
                model_string += "beta[{0}] ~ {1};\n".format(i+1, prior.to_string()) # stan is 1-indexed
        return model_string


    def stan_code(self, beta_priors):
        return load_model_template().format(y_type=self.distribution.y_type,
                                            parameter_statement=self.distribution.parameter_statement,
                                            link_function=self.link,
                                            model_statement=self.distribution.model_statement,
                                            beta_priors=self.parameter_priors_to_string(beta_priors))


class RegressionModels:
    linear = RegressionModel(OutcomeDistributions.gaussian, "")
    logistic  = RegressionModel(OutcomeDistributions.bernoulli_logit, "")
    logistic_less_efficient  = RegressionModel(OutcomeDistributions.bernoulli, "inv_logit")
    probit = RegressionModel(OutcomeDistributions.bernoulli, "Phi")
    probit_approx = RegressionModel(OutcomeDistributions.bernoulli, "Phi_approx")


@dispatch(np.ndarray, np.ndarray, RegressionModel)
def glm(x, y, regression_type, iterations = 100, priors = []):
    N, k = x.shape
    model_code = regression_type.stan_code(priors)
    print(model_code)
    fit = stan_cache(model_code=model_code, data={"x": x, "N": N, "K": k, "y": y}, iter=iterations, chains=4)
    return fit

def slice_to_range(s):
    return range(s.start, s.stop)

@dispatch(str, object, RegressionModel)
def glm(formula, df, regression_type, priors = {}):
    y, x = dmatrices(formula, df)
    x_ = np.asarray(x)
    if regression_type.distribution.y_type == "int": # (necessary b/c patsy converts to float)
        y = y.astype("int")
    y_ = np.asarray(np.squeeze(y))
    print(type(y_[0]))
    beta_priors_list = [(slice_to_range(x.design_info.slice(key)), val) for key, val in priors.items()]
    return glm(x_, y_, regression_type, priors= beta_priors_list)