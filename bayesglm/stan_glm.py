import os

from patsy import dmatrices
import multipledispatch
import numpy as np

from bayesglm.stan_cache import stan_cache


def load_model_template():
    stan_model_path = os.path.join(os.path.dirname(__file__), "model.stan")
    with open(stan_model_path, "r") as stan_model_file:
        stan_model_template = stan_model_file.read()
    return stan_model_template


def parameter_priors_to_string(parameter_priors):
    # parameter_priors should be a list of tuples (r, prior) where r is a range and prior is a prior like NormalPrior
    # This approach leads to different stan models depending on which parameters you choose to give which priors.
    # We could probably avoid recompiling so much by switching to an approach where the stan code is always the same
    #   for any set of priors.
    model_string = ""
    for r, prior in parameter_priors:
        for i in r:
            model_string += "beta[{0}] ~ {1};\n".format(i + 1, prior.to_string())  # stan is 1-indexed
    return model_string


def stan_code(family, beta_priors):
    return load_model_template().format(y_type=family.distribution.y_type,
                                        parameter_statement=family.distribution.parameter_statement,
                                        link_function=family.link,
                                        model_statement=family.distribution.model_statement,
                                        beta_priors=parameter_priors_to_string(beta_priors))


@multipledispatch.dispatch(np.ndarray, np.ndarray)
def stan_glm(x, y, family, iterations=2000, priors=(), **kwargs):
    num_rows, num_predictors = x.shape
    model_code = stan_code(family, priors)
    fit = stan_cache(model_code=model_code,
                     data={"x": x, "N": num_rows, "K": num_predictors, "y": y},
                     iter=iterations,
                     chains=4,
                     **kwargs)
    return fit


@multipledispatch.dispatch(str, object)
def stan_glm(formula, df, family, priors=None, **kwargs):
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
    return stan_glm(x_, y_, family=family, priors=beta_priors_list, **kwargs)