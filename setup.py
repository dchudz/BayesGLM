from distutils.core import setup

setup(
    name='BayesGLM',
    version='0.1dev',
    install_requires=['pystan', 'patsy', 'multipledispatch', 'pandas'],
    extras_require={'dev': ['ipython', 'rpy2', 'nose', 'scipy']},
    packages=['bayesglm'],
)
