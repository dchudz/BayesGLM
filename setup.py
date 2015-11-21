from distutils.core import setup

setup(
    name='BayesGLM',
    version='0.1dev',
    install_requires = ['pystan', 'patsy', 'multipledispatch', 'pandas'],
    packages=['bayesglm',],
)
