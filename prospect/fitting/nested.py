import sys
import numpy as np
from numpy.random import normal, multivariate_normal

try:
    import nestle
except(ImportError):
    pass

__all__ = ["run_nestle_sampler"]


def run_nestle_sampler(lnprobfn, model, verbose=True,
                       callback=nestle.print_progress,
                       nestle_method='single', nestle_npoints=200,
                       nestle_maxcall=int(1e6), nestle_update_interval=None,
                       **kwargs):

    result = nestle.sample(lnprobfn, model.prior_transform, model.ndim,
                           method=nestle_method, npoints=nestle_npoints,
                           callback=callback, maxcall=nestle_maxcall,
                           update_interval=nestle_update_interval)
    return result
