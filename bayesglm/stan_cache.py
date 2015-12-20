# https://pystan.readthedocs.org/en/latest/avoiding_recompilation.html

import pickle
import pystan
from hashlib import md5
import bayesglm
import os


def stan_cache(model_code, model_name=None, **kwargs):
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    cache_dir = os.path.join(bayesglm.__path__[0], ".cached_models")
    try:
        os.makedirs(cache_dir)
    except OSError:
        if not os.path.isdir(cache_dir):
            raise
    if model_name is None:
        cache_fn = os.path.join(cache_dir, 'cached-model-{}.pkl'.format(code_hash))
    else:
        cache_fn = os.path.join(cache_dir, 'cached-{}-{}.pkl'.format(model_name, code_hash))
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm.sampling(**kwargs)
