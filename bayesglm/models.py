from bayesglm.stan_cache import stan_cache
from collections import namedtuple
from patsy import dmatrices
from multipledispatch import dispatch
import numpy as np


MODEL_CODE = """
   data {{
     int<lower=1> K;
     int<lower=0> N;
     {0} y[N];
     matrix[N,K] x;
   }}
   parameters {{
     vector[K] beta;
     {1}
   }}
   model {{
    real mu[N];
    vector[N] eta   ;
    eta <- x*beta;
    for (i in 1:N) {{
       mu[i] <- {2}(eta[i]);
    }};
    {3}
    {4}
   }}
"""



# x_.design_info.slice("c[T.c]")

# maybe priors are specified like:
#   iid_priors=Normal(0,10)
#   iid_priors=Uniform(5,15)

# "y ~ x1 + x2"

# {"x1": Uniform(5, 15)}
# {"x2": Normal(0,10)}
# {"cat1": Normal(0,1)} - refers to all
# {"1": (constant term)

# nothing: uniform improper

# can we include all supported priors in the loop to have fewer models?

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
        return MODEL_CODE.format(self.distribution.y_type,
                                 self.distribution.parameter_statement,
                                 self.link,
                                 self.distribution.model_statement,
                                 self.parameter_priors_to_string(beta_priors))


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