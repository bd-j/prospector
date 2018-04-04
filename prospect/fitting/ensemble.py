import sys
import numpy as np
from numpy.random import normal, multivariate_normal

try:
    import emcee
    EMCEE_VERSION = emcee.__version__.split('.')[0]
except(ImportError):
    pass

from ..models.priors import plotting_range
from .convergence import convergence_check

__all__ = ["run_emcee_sampler", "reinitialize_ball", "sampler_ball",
           "emcee_burn"]


def run_emcee_sampler(lnprobfn, initial_center, model, verbose=True,
                      postargs=[], postkwargs={}, prob0=None,
                      nwalkers=None, nburn=[16], niter=32,
                      walker_factor=4, storechain=True,
                      pool=None, hdf5=None, interval=1,
                      convergence_check_interval=None, convergence_chunks=325,
                      convergence_stable_points_criteria=3,
                      **kwargs):
    """Run an emcee sampler, including iterations of burn-in and re -
    initialization.  Returns the production sampler.

    :param lnprobfn:
        The posterior probability function.

    :param initial_center:
        The initial center for the sampler ball

    :param model:
        An instance of a models.ProspectorParams object.

    :param postargs:
        Positional arguments for ``lnprobfn``.

    :param postkwargs:
        Keyword arguments for ``lnprobfn``.

    :param nwalkers:
        The number of walkers to use.  If None, use the nearest power of two to
        ``ndim * walker_factor``.

    :param niter:
        Number of iterations for the production run

    :param nburn:
        List of the number of iterations to run in each round of burn-in (for
        removing stuck walkers.) E.g. `nburn=[32, 64]` will run the sampler for
        32 iterations before reinitializing and then run the sampler for
        another 64 iterations before starting the production run.

    :param storechain: (default: True)
        If using HDF5 output, setting this to False will keep the chain from
        being held in memory by the sampler object.

    :param pool: (optional)
        A ``Pool`` object, either from ``multiprocessing`` or from
        ``emcee.mpi_pool``.

    :param hdf5: (optional)
        H5py.File object that will be used to store the chain in the datasets
        ``"chain"`` and ``"lnprobability"``.  If not set, the chain will instead
        be stored as a numpy array in the returned sampler object

    :param interval:
        Fraction of the full run at which to flush to disk, if using hdf5 for
        output.

    :param convergence_check_interval:
        How often to assess convergence, in number of iterations. If this is
        set, then the KL convergence test is run.

    :param convergence_chunks:
        The number of iterations to combine when creating the marginalized
        parameter probability functions.

    :param convergence_stable_points_criteria:
        The number of stable convergence checks that the chain must pass before
        being declared stable.
    """
    # Get dimensions
    ndim = model.ndim
    if nwalkers is None:
        nwalkers = int(2 ** np.round(np.log2(ndim * walker_factor)))
    if verbose:
        print('number of walkers={}'.format(nwalkers))

    # Initialize sampler
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn,
                                     args=postargs, kwargs=postkwargs,
                                     pool=pool)
    # Burn in sampler
    initial, in_cent, in_prob = emcee_burn(esampler, initial_center, nburn, model,
                                           prob0=prob0, verbose=verbose, **kwargs)
    # Production run
    esampler.reset()

    if hdf5 is not None:
        # Set up hdf5 backend
        sdat = hdf5.create_group('sampling')
        if convergence_check_interval is None:
            # static dataset
            chain  = sdat.create_dataset("chain", (nwalkers, niter, ndim))
            lnpout = sdat.create_dataset("lnprobability", (nwalkers, niter))
        else:
            # dynamic dataset
            conv_int = convergence_check_interval
            conv_crit = convergence_stable_points_criteria
            nfirstcheck = (2 * convergence_chunks +  conv_int * (conv_crit - 1))

            chain   = sdat.create_dataset('chain',
                                          (nwalkers, nfirstcheck, ndim),
                                          maxshape=(nwalkers, None, ndim))
            lnpout  = sdat.create_dataset('lnprobability',
                                          (nwalkers, nfirstcheck),
                                          maxshape=(nwalkers, None))
            kl      = sdat.create_dataset('kl_divergence',
                                          (conv_crit, ndim),
                                          maxshape=(None, ndim))
            kl_iter = sdat.create_dataset('kl_iteration',
                                          (conv_crit,),
                                          maxshape=(None,))
    else:
        storechain = True

    # Do some emcee version specific choices
    if EMCEE_VERSION == '3':
        mc_args = {"store": storechain,
                   "iterations": niter}
    else:
        mc_args = {"storechain": storechain,
                   "iterations": niter}

    # Main loop over iterations of the MCMC sampler
    if verbose:
        print('starting production')
    for i, result in enumerate(esampler.sample(initial, **mc_args)):
        if hdf5 is not None:
            chain[:, i, :] = result[0]
            lnpout[:, i] = result[1]
            do_convergence_check = ((convergence_check_interval is not None) and
                                    (i+1 >= nfirstcheck) and
                                    ((i+1 - nfirstcheck) % convergence_check_interval == 0))
            if do_convergence_check:
                if verbose:
                    print('checking convergence after iteration {0}').format(i+1)
                converged, info = convergence_check(chain,
                                                    convergence_check_interval=conv_int,
                                                    convergence_stable_points_criteria=conv_crit,
                                                    convergence_chunks=convergence_chunks, **kwargs)
                kl[:, :] = info['kl_test']
                kl_iter[:] = info['iteration']
                hdf5.flush()

                if converged:
                    if verbose:
                        print('converged, ending emcee.')
                    break
                else:
                    if verbose:
                        print('not converged, continuing.')
                    if (i+1 > (niter-convergence_check_interval)):
                        # if we're going to exit soon, do something fancy
                        ngrow = niter - (i + 1)
                        chain.resize(chain.shape[1]+ngrow, axis=1)
                        lnpout.resize(lnpout.shape[1]+ngrow, axis=1)
                    else:
                        # else extend by convergence_check_interval
                        chain.resize(chain.shape[1] + convergence_check_interval, axis=1)
                        lnpout.resize(lnpout.shape[1] + convergence_check_interval, axis=1)
                        kl.resize(kl.shape[0] + 1, axis=0)
                        kl_iter.resize(kl_iter.shape[0] + 1, axis=0)

            if (np.mod(i+1, int(interval*niter)) == 0) or (i+1 == niter):
                # do stuff every once in awhile
                # this would be the place to put some callback functions
                # e.g. [do(result, i, esampler) for do in things_to_do]
                # like, should probably store the random state too.
                hdf5.flush()
    if verbose:
        print('done production')

    return esampler, in_cent, in_prob


def emcee_burn(sampler, initial_center, nburn, model=None, prob0=None,
               initial_disp=0.1, verbose=True, **kwargs):
    """Run the emcee sampler for nburn iterations, reinitializing after each
    round.

    :param nburn:
        List giving the number of iterations in each round of burn-in.
        E.g. nburn=[32, 64] will run the sampler for 32 iterations before
        reinitializing and then run the sampler for another 64 iterations
    """
    # Do some emcee version specific choices
    if EMCEE_VERSION == '3':
        nwalkers = sampler.nwalkers
        mc_args = {"store": True}
    else:
        nwalkers = sampler.k
        mc_args = {"storechain": True}

    # Set up initial positions
    model.set_parameters(initial_center)
    disps = model.theta_disps(default_disp=initial_disp)
    limits = np.array(model.theta_bounds()).T
    if hasattr(model, 'theta_disp_floor'):
        disp_floor = model.theta_disp_floor()
    else:
        disp_floor = 0.0
    disps = np.sqrt(disps**2 + disp_floor**2)
    initial = resample_until_valid(sampler_ball, initial_center, disps, nwalkers,
                                   limits=limits, prior_check=model)

    # Start the burn-in
    for k, iburn in enumerate(nburn):
        epos, eprob, state = sampler.run_mcmc(initial, iburn, **mc_args)
        # Find best walker position
        best = sampler.flatlnprobability.argmax()
        # Is new position better than old position?
        if prob0 is None or sampler.flatlnprobability[best] > prob0:
            prob0 = sampler.flatlnprobability[best]
            initial_center = sampler.flatchain[best, :]
        if k == len(nburn):
            # Done burning.
            if verbose:
                print('done all burn-in.')
            # Don't construct new sampler ball after last burn-in.
            sampler.reset()
            continue
        if epos.shape[0] < model.ndim*2:
            initial = reinitialize_ball(epos, eprob, center=initial_center,
                                        limits=limits, disp_floor=disp_floor,
                                        prior_check=model, **kwargs)
        else:
            initial = reinitialize_ball_covar(epos, eprob, center=initial_center,
                                              limits=limits, disp_floor=disp_floor,
                                              prior_check=model, **kwargs)
        sampler.reset()
        if verbose:
            print('done burn #{}'.format(k))

    return epos, initial_center, prob0


def reinitialize_ball_covar(pos, prob, threshold=50.0, center=None,
                            disp_floor=0.0, **extras):
    """Estimate the parameter covariance matrix from the positions of a
    fraction of the current ensemble and sample positions from the multivariate
    gaussian corresponding to that covariance matrix.  If ``center`` is not
    given the center will be the mean of the (fraction of) the ensemble.

    :param pos:
        The current positions of the ensemble, ndarray of shape (nwalkers, ndim)

    :param prob:
        The current probabilities of the ensemble, used to reject some fraction
        of walkers with lower probability (presumably stuck walkers).  ndarray
        of shape (nwalkers,)

    :param threshold: default 50.0
        Float in the range [0,100] giving the fraction of walkers to throw away
        based on their ``prob`` before estimating the covariance matrix.

    :param center: optional
        The center of the multivariate gaussian. If not given or ``None``, then
        the center will be estimated from the mean of the postions of the
        acceptable walkers.  ndarray of shape (ndim,)

    :param limits: optional
        An ndarray of shape (2, ndim) giving lower and upper limits for each
        parameter.  The newly generated values will be clipped to these limits.
        If the result consists only of the limit then a vector of small random
        numbers will be added to the result.

    :returns pnew:
        New positions for the sampler, ndarray of shape (nwalker, ndim)
    """
    pos = np.atleast_2d(pos)
    nwalkers = prob.shape[0]
    good = prob > np.percentile(prob, threshold)
    if center is None:
        center = pos[good, :].mean(axis=0)
    Sigma = np.cov(pos[good, :].T)
    Sigma[np.diag_indices_from(Sigma)] += disp_floor**2
    pnew = resample_until_valid(multivariate_normal, center, Sigma,
                                nwalkers, **extras)
    return pnew


def reinitialize_ball(pos, prob, center=None, ptiles=[25, 50, 75],
                      disp_floor=0., **extras):
    """Choose the best walker and build a ball around it based on the other
    walkers.  The scatter in the new ball is based on the interquartile range
    for the walkers in their current positions
    """
    pos = np.atleast_2d(pos)
    nwalkers = pos.shape[0]
    if center is None:
        center = pos[prob.argmax(), :]
    tmp = np.percentile(pos, ptiles, axis=0)
    # 1.35 is the ratio between the 25-75% interquartile range and 1
    # sigma (for a normal distribution)
    scatter = np.abs((tmp[2] - tmp[0]) / 1.35)
    scatter = np.sqrt(scatter**2 + disp_floor**2)

    pnew = resample_until_valid(sampler_ball, initial_center, scatter,
                                nwalkers, **extras)
    return pnew


def resample_until_valid(sampling_function, center, sigma, nwalkers,
                         limits=None, maxiter=1e3, prior_check=None, **extras):
    """Sample from the sampling function, with optional clipping to prior
    bounds and resampling in the case of parameter positions that are outside
    complicated custom priors.

    :param sampling_function:
        The sampling function to use, it must have the calling sequence
        ``sampling_function(center, sigma, size=size)``

    :param center:
        The center of the distribution

    :param sigma:
        Array describing the scatter of the distribution in each dimension.
        Can be two-dimensional, e.g. to describe a covariant multivariate
        normal (if the sampling function takes such a thing).

    :param nwalkers:
        The number of valid samples to produce.

    :param limits: (optional)
        Simple limits on the parameters, passed to ``clip_ball``.

    :param prior_check: (optional)
        An object that has a ``prior_product()`` method which returns the prior
        ln(probability) for a given parameter position.

    :param maxiter:
        Maximum number of iterations to try resampling before giving up and
        returning a set of parameter positions at least one of which is not
        within the prior.

    :returns pnew:
        New parameter positions, ndarray of shape (nwalkers, ndim)
    """
    invalid = np.ones(nwalkers, dtype=bool)
    pnew = np.zeros([nwalkers, len(center)])
    for i in range(int(maxiter)):
        # replace invalid elements with new samples
        tmp = sampling_function(center, sigma, size=invalid.sum())
        pnew[invalid, :] = tmp
        if limits is not None:
            # clip to simple limits
            if sigma.ndim > 1:
                diag = np.diag(sigma)
            else:
                diag = sigma
            pnew = clip_ball(pnew, limits, diag)
        if prior_check is not None:
            # check the prior
            lnp = np.array([prior_check.prior_product(pos) for pos in pnew])
            invalid = ~np.isfinite(lnp)
            if invalid.sum() == 0:
                # everything is valid, return
                return pnew
        else:
            # No prior check, return on first iteration
            return pnew
    # reached maxiter, return whatever exists so far
    print("initial position resampler hit ``maxiter``")
    return pnew


def sampler_ball(center, disp, size=1):
    """Produce a ball around a given position.  This should probably be a
    one-liner.
    """
    ndim = center.shape[0]
    if np.size(disp) == 1:
        disp = np.zeros(ndim) + disp
    pos = normal(size=[size, ndim]) * disp[None, :] + center[None, :]
    return pos


def clip_ball(pos, limits, disp):
    """Clip to limits.  If all samples below (above) limit, add (subtract) a
    uniform random number (scaled by ``disp``) to the limit.
    """
    npos = pos.shape[0]
    pos = np.clip(pos, limits[0][None, :], limits[1][None, :])
    for i, p in enumerate(pos.T):
        u = np.unique(p)
        if len(u) == 1:
            tiny = disp[i] * np.random.uniform(0, disp[i], npos)
            if u == limits[0, i]:
                pos[:, i] += tiny
            if u == limits[1, i]:
                pos[:, i] -= tiny
    return pos


def restart_sampler(sample_results, lnprobf, sps, niter,
                    nthreads=1, pool=None):
    """Restart a sampler from its last position and run it for a specified
    number of iterations.  The sampler chain and the model object should be
    given in the sample_results dictionary.  Note that lnprobfn and sps must be
    defined at the global level in the same way as the sampler originally ran,
    or your results will be super weird (and wrong)!

    Unimplemented/tested
    """
    model = sample_results['model']
    initial = sample_results['chain'][:, -1, :]
    nwalkers, ndim = initial.shape
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf, args=[model],
                                     threads=nthreads,  pool=pool)
    epos, eprob, state = esampler.run_mcmc(initial, niter, rstate0=state)
    return esampler
