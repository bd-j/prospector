import sys, time
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


def run_dynesty_sampler(lnprobfn, prior_transform, ndim, verbose=True,
                        nested_bound='multi', nested_sample='rwalk',
                        nested_nlive=100, nested_live_points=None,
                        nested_nlive_init=100, nested_nlive_batch=100,
                        nested_update_interval=0.6, nested_walks=25,
                        nested_maxcall=None, nested_maxiter=None,
                        pool=None, queue_size=1, nested_use_stop=True,
                        nested_maxbatch=None, nested_weight_kwargs=None,
                        nested_bootstrap=None, nested_dlogz_init=0.01,
                        use_pool={}, **kwargs):

    dsampler = dynesty.DynamicNestedSampler(lnprobfn, prior_transform, ndim,
                                            bound=nested_bound, sample=nested_sample,
                                            update_interval=nested_update_interval,
                                            pool=pool, queue_size=queue_size,
                                            nlive=nested_nlive, walks=nested_walks,
                                            bootstrap=nested_bootstrap,use_pool=use_pool)
                                     
    # run dynesty
    tstart = time.time()
    try:
        dsampler.run_nested(nlive_init=nested_nlive_init,
                            dlogz_init=nested_dlogz_init,
                            maxbatch=nested_maxbatch, 
                            nlive_batch=nested_nlive_batch,
                            use_stop=nested_use_stop, wt_kwargs=nested_weight_kwargs)
    except:
        import pickle
        print 'crashed! dumping output in crash_dns.pkl'
        with open('crash_dns.pkl', 'w') as f:
            pickle.dump(dsampler.results, f)
        sys.exit()

    dresult = dsampler.results
    ndur = time.time() - tstart

    if verbose:
        print('done dynesty in {0}s'.format(ndur))

    #for it, s in enumerate(nsampler.sample(maxiter=nested_maxiter, maxcall=nested_maxcall)):
    #    pass
    #blob = nsampler.add_live_points()
    
    return dresult

