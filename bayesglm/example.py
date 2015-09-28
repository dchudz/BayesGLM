import numpy as np
from scipy.special import logit, expit
import pandas as pd
from bayesglm import *

N = 200
x = np.random.normal(size=(N,2))
beta = np.array([15,5])
y = np.dot(x, beta) + np.random.normal(size=N)
y_binary = np.random.binomial(n=1, p=expit(y))

df = pd.DataFrame(x, columns = ["x1", "x2"])
df['y'] = y
df['y_binary'] = y_binary

# fit = glm("y ~ 0 + x1 + x2", df, RegressionModels.linear, priors = {"x1": NormalPrior(0,.1),
#                                                                          "x2": NormalPrior(50,1)})

#fit = glm("y ~ 0 + x1 + x2", df, RegressionModels.linear)

fit = glm("y ~ x1 + x2", df, RegressionModels.linear, priors = {"Intercept": NormalPrior(10,.1)})


print(fit)