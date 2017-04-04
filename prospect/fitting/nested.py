import sys
import numpy as np
from numpy.random import normal, multivariate_normal

try:
    import nestle
except(ImportError):
    pass

__all__ = ["run_nestle_sampler"]

def run_nestle_sampler(lnprobfn, model, verbose=True, callback=None,
                       nestle_method='single', nestle_npoints=200,
                       **kwargs):


    result = nestle.sample(lnprobfn, model.prior_transform, model.ndim,
                           method=nestle_method, npoints=nestle_npoints,
                           callback=nestle.print_progress, maxcall=int(1e6))
    return result
