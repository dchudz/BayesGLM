import numpy as np
from scipy.special import logit, expit
import pandas as pd

N = 200
x = np.random.normal(size=(N,2))
beta = np.array([15,5])
y = np.dot(x, beta) + np.random.normal(size=N)
y_binary = np.random.binomial(n=1, p=expit(y))

df = pd.DataFrame(x, columns = ["x1", "x2"])

df['y'] = y
df['y_binary'] = y_binary