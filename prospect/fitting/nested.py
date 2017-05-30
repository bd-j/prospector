import sys
import numpy as np
from numpy.random import normal, multivariate_normal

try:
    import nestle
except(ImportError):
    pass


try:
    import dynesty
except(ImportError):
    pass


__all__ = ["run_nestle_sampler", "run_dynesty_sampler"]


def run_nestle_sampler(lnprobfn, model, verbose=True,
                       callback=None,
                       nestle_method='single', nestle_npoints=200,
                       nestle_maxcall=int(1e6), nestle_update_interval=None,
                       **kwargs):

    result = nestle.sample(lnprobfn, model.prior_transform, model.ndim,
                           method=nestle_method, npoints=nestle_npoints,
                           callback=callback, maxcall=nestle_maxcall,
                           update_interval=nestle_update_interval)
    return result


def run_dynesty_sampler(lnprobfn, model, verbose=True,
                        nested_method='multi', nested_sample='unif',
                        nested_nlive=200, nested_live_points=None,
                        nested_update_interval=None,
                        nested_maxcall=int(1e6), nested_maxiter=int(1e6),
                        pool=None, queue_size=1, **kwargs):

    #if pool is not None:
    #    queue_size = pool._max_workers
    #else:
    #    queue_size = 1
    
    nsampler = dynesty.NestedSampler(lnprobfn, model.prior_transform, model.ndim,
                                     nlive=nested_nlive,
                                     bound=nested_method, sample=nested_sample,
                                     pool=pool, queue_size=queue_size)
                                     
    for it, s in enumerate(nsampler.sample(maxiter=nested_maxiter, maxcall=nested_maxcall)):
        pass
    blob = nsampler.add_live_points()
    
    return nsampler
