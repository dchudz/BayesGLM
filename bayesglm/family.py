from collections import namedtuple


OutcomeDistribution = namedtuple('OutcomeDistribution', ['y_type',
                                                         'parameter_names',
                                                         'parameter_statement',
                                                         'model_statement'])


class Family(object):
    def __init__(self, distribution, link = None):
        self.distribution = distribution
        self.link = link


def gaussian(link = ""):
    outcome_distribution = OutcomeDistribution(
        y_type="real",
        parameter_names=["sigma"],
        parameter_statement="real sigma;",
        model_statement="y ~ normal(mu, sigma);")
    return Family(outcome_distributio]n, link=link)


def bernoulli_logit():
    # Stan provides "bernouli_logit" distribution, combining logit link function w/ bernouli distribution.
    # (this is more efficient than separating them)
    outcome_distribution = OutcomeDistribution(
        y_type="int",
        parameter_names=[],
        parameter_statement="",
        model_statement="y ~ bernoulli_logit(mu);")
    return Family(outcome_distribution, link="")


def bernoulli(link = "inv_logit"):
    # less efficient than bernouli_logit
    outcome_distribution = OutcomeDistribution(
        y_type="int",
        parameter_names=[],
        parameter_statement="",
        model_statement="y ~ bernoulli(mu);")
    return Family(outcome_distribution, link=link)


def probit():
    return bernoulli(link="Phi")


def probit_approx():
    return bernoulli("Phi_approx")
