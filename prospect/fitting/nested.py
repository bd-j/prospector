import sys, time
import numpy as np
from numpy.random import normal, multivariate_normal
from six.moves import range

try:
    import nestle
except(ImportError):
    pass


try:
    import dynesty
    from dynesty.utils import *
    from dynesty.dynamicsampler import _kld_error
except(ImportError):
    pass


__all__ = ["run_nestle_sampler", "run_dynesty_sampler"]


def run_nestle_sampler(lnprobfn, model, verbose=True,
                       callback=None,
                       nestle_method='multi', nestle_npoints=200,
                       nestle_maxcall=int(1e6), nestle_update_interval=None,
                       **kwargs):

    result = nestle.sample(lnprobfn, model.prior_transform, model.ndim,
                           method=nestle_method, npoints=nestle_npoints,
                           callback=callback, maxcall=nestle_maxcall,
                           update_interval=nestle_update_interval)
    return result


def run_dynesty_sampler(lnprobfn, prior_transform, ndim,
                        verbose=True,
                        # sampler kwargs
                        nested_bound='multi',
                        nested_sample='unif',
                        nested_walks=25,
                        nested_update_interval=0.6,
                        nested_bootstrap=0,
                        pool=None,
                        use_pool={},
                        queue_size=1,
                        # init sampling kwargs
                        nested_nlive_init=100,
                        nested_dlogz_init=0.02,
                        nested_maxiter_init=None,
                        nested_maxcall_init=None,
                        nested_live_points=None,
                        # batch sampling kwargs
                        nested_maxbatch=None,
                        nested_nlive_batch=100,
                        nested_maxiter_batch=None,
                        nested_maxcall_batch=None,
                        nested_use_stop=True,
                        # overall kwargs
                        nested_maxcall=None,
                        nested_maxiter=None,
                        nested_first_update={},
                        stop_function=None,
                        wt_function=None,
                        nested_weight_kwargs={'pfrac': 1.0},
                        nested_stop_kwargs={},
                        nested_save_bounds=False,
                        print_progress=True,
                        **extras):

    # instantiate sampler
    dsampler = dynesty.DynamicNestedSampler(lnprobfn, prior_transform, ndim,
                                            bound=nested_bound,
                                            sample=nested_sample,
                                            walks=nested_walks,
                                            bootstrap=nested_bootstrap,
                                            update_interval=nested_update_interval,
                                            pool=pool, queue_size=queue_size, use_pool=use_pool
                                            )

    # generator for initial nested sampling
    ncall = dsampler.ncall
    niter = dsampler.it - 1
    tstart = time.time()
    for results in dsampler.sample_initial(nlive=nested_nlive_init,
                                           dlogz=nested_dlogz_init,
                                           maxcall=nested_maxcall_init,
                                           maxiter=nested_maxiter_init,
                                           live_points=nested_live_points):

        try:
            # dynesty >= 2.0
            (worst, ustar, vstar, loglstar, logvol,
             logwt, logz, logzvar, h, nc, worst_it,
             propidx, propiter, eff, delta_logz, blob) = results
        except(ValueError):
            # dynsety < 2.0
            (worst, ustar, vstar, loglstar, logvol,
            logwt, logz, logzvar, h, nc, worst_it,
            propidx, propiter, eff, delta_logz) = results

        if delta_logz > 1e6:
            delta_logz = np.inf
        ncall += nc
        niter += 1

        if print_progress:
            with np.errstate(invalid='ignore'):
                logzerr = np.sqrt(logzvar)
            sys.stderr.write("\riter: {:d} | batch: {:d} | nc: {:d} | "
                            "ncall: {:d} | eff(%): {:6.3f} | "
                            "logz: {:6.3f} +/- {:6.3f} | "
                            "dlogz: {:6.3f} > {:6.3f}    "
                            .format(niter, 0, nc, ncall, eff, logz,
                                    logzerr, delta_logz, nested_dlogz_init))
            sys.stderr.flush()

    ndur = time.time() - tstart
    if verbose:
        print('\ndone dynesty (initial) in {0}s'.format(ndur))

    if nested_maxcall is None:
        nested_maxcall = sys.maxsize
    if nested_maxbatch is None:
        nested_maxbatch = sys.maxsize
    if nested_maxcall_batch is None:
        nested_maxcall_batch = sys.maxsize
    if nested_maxiter is None:
        nested_maxiter = sys.maxsize
    if nested_maxiter_batch is None:
        nested_maxiter_batch = sys.maxsize

    # generator for dynamic sampling
    tstart = time.time()
    for n in range(dsampler.batch, nested_maxbatch):
        # Update stopping criteria.
        dsampler.sampler.save_bounds = False
        res = dsampler.results
        mcall = min(nested_maxcall - ncall, nested_maxcall_batch)
        miter = min(nested_maxiter - niter, nested_maxiter_batch)
        if nested_use_stop:
            if dsampler.use_pool_stopfn:
                M = dsampler.M
            else:
                M = map
            stop, stop_vals = stop_function(res, nested_stop_kwargs,
                                            rstate=dsampler.rstate, M=M,
                                            return_vals=True)
            stop_val = stop_vals[2]
        else:
            stop = False
            stop_val = np.NaN

        # If we have either likelihood calls or iterations remaining,
        # run our batch.
        if mcall > 0 and miter > 0 and not stop:
            # Compute our sampling bounds using the provided
            # weight function.
            logl_bounds = wt_function(res, nested_weight_kwargs)
            lnz, lnzerr = res.logz[-1], res.logzerr[-1]
            for results in dsampler.sample_batch(nlive_new=nested_nlive_batch,
                                                 logl_bounds=logl_bounds,
                                                 maxiter=miter,
                                                 maxcall=mcall,
                                                 save_bounds=nested_save_bounds):

                try:
                    # dynesty >= 2.0
                    (worst, ustar, vstar, loglstar, nc,
                     worst_it, propidx, propiter, eff, blob) = results
                except(ValueError):
                    # dynesty < 2.0
                    (worst, ustar, vstar, loglstar, nc,
                    worst_it, propidx, propiter, eff) = results
                ncall += nc
                niter += 1
                if print_progress:
                    sys.stderr.write("\riter: {:d} | batch: {:d} | "
                                    "nc: {:d} | ncall: {:d} | "
                                    "eff(%): {:6.3f} | "
                                    "loglstar: {:6.3f} < {:6.3f} "
                                    "< {:6.3f} | "
                                    "logz: {:6.3f} +/- {:6.3f} | "
                                    "stop: {:6.3f}    "
                                    .format(niter, n+1, nc, ncall,
                                            eff, logl_bounds[0], loglstar,
                                            logl_bounds[1], lnz, lnzerr,
                                            stop_val))
                    sys.stderr.flush()
            dsampler.combine_runs()
        else:
            # We're done!
            break

    ndur = time.time() - tstart
    if verbose:
        print('done dynesty (dynamic) in {0}s'.format(ndur))

    return dsampler.results

