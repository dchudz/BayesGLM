# BayesGLM

This package aims to bring convenient Bayesian GLMs (and other GLM-like models) to Python, with the help of [Patsy](https://patsy.readthedocs.org/en/latest/), [Stan](http://mc-stan.org/), and [PyStan](https://pystan.readthedocs.org/en/latest/).

Here are some examples:

```
from bayesglm import *
from bayesglm.example import df

# linear regression
glm("y ~ x1 + x2", df, RegressionModels.linear)

# linear regression with no intercept
glm("y ~ 0 + x1 + x2", df, RegressionModels.linear)

# linear regression w/ priors
glm("y ~ x1 + x2", df, RegressionModels.linear, priors = {"Intercept": NormalPrior(0, 10), "x2": NormalPrior(0,5)})

# flat priors for main effects, normal prior for interaction term
# (in patsy's notation, "x1*x2" includes the interaction and main effects. "x1:x2" refers to the interaction term.)
glm("y ~ x1*x2", df, RegressionModels.linear, priors = {"x1:x2": NormalPrior(0,5)})

# logistic regression
glm("y_binary ~ x1 + x2", df, RegressionModels.logistic)

# specify the outcome distribution and link function by hand:
glm("y_binary ~ x1 + x2", df, RegressionModel(OutcomeDistributions.bernoulli, "Phi"))
```

Todo:

- give better output
	+ (At the moment, `glm` gives you raw output from PyStan. I should do something to translate that back into the terms of the original formula / dataframe.)
- allow making predictions on new data
- ability to specify prior for parameters of the outcome distribution (e.g. `sigma` for normal distributions)
- switch the way I specify `beta`s priors in the Stan code so that we don't have to compile so many models. (e.g. the parameters of the prior distribution should be data sent to Stan, not in the Stan code itself.)
- prepackage compiled Stan models so that user doesn't have to wait for compilation even once

After that:

- hierarchical models!
