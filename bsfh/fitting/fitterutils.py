import sys
import numpy as np
from numpy.random import normal, multivariate_normal
import emcee
from . import minimizer
from ..models.priors import plotting_range

__all__ = ["run_emcee_sampler", "reinitialize_ball", "sampler_ball", "emcee_burn",
           "pminimize", "minimizer_ball", "reinitialize"]


def run_emcee_sampler(lnprobf, initial_center, model, verbose=True,
                      postargs=[], postkwargs={}, prob0=None,
                      nwalkers=None, nburn=[16], niter=32,
                      walker_factor=4, initial_disp=0.1,
                      nthreads=1, pool=None, hdf5=None, interval=1,
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
        List of the number of iterations to run in each round of brun-in (for
        removing stuck walkers)

    :param pool: (optional)
        A ``Pool`` object, either from ``multiprocessing`` or from
        ``emcee.mpi_pool``.

    :param hdf5: (optional)
        H5py.File object that will be used to store the chain in the datasets
        ``"chain"`` and ``"lnprobability"``.  If not set, the chin will instead
        be stored as a numpy array in the returned sampler object

    :param interval:
        Fraction of the full run at which to flush to disk, if using hdf5 for
        output.
    """
    # Get dimensions
    ndim = model.ndim
    if nwalkers is None:
        nwalkers = int(2 ** np.round(np.log2(ndim * walker_factor)))
    if verbose:
        print('number of walkers={}'.format(nwalkers))

    # Set up initial positions
    model.set_parameters(initial_center)
    disps = model.theta_disps(default_disp=initial_disp)
    limits = np.array(model.theta_bounds()).T
    if hasattr(model, 'theta_disp_floor'):
        disps = np.sqrt(disps**2 + model.theta_disp_floor()**2)
    initial = resample_until_valid(sampler_ball, initial_center, disps, nwalkers,
                                   limits=limits, prior_check=model)

    # Initialize sampler
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf,
                                     args=postargs, kwargs=postkwargs,
                                     threads=nthreads, pool=pool)
    # Burn in sampler
    initial, in_cent, in_prob = emcee_burn(esampler, initial, nburn, model,
                                           prob0=prob0, verbose=verbose, **kwargs)
    # Production run
    esampler.reset()
    if hdf5 is not None:
        # Set up output
        chain = hdf5.create_dataset("chain", (nwalkers, niter, ndim))
        lnpout = hdf5.create_dataset("lnprobability", (nwalkers, niter))
        # blob = hdf5.create_dataset("blob")
        storechain = False
    else:
        storechain = True

    # Main loop over iterations of the MCMC sampler
    for i, result in enumerate(esampler.sample(initial, iterations=niter,
                                               storechain=storechain)):
        if hdf5 is not None:
            chain[:, i, :] = result[0]
            lnpout[:, i] = result[1]
            if np.mod(i+1, int(interval*niter)) == 0:
                # do stuff every once in awhile
                # this would be the place to put some callback functions
                # e.g. [do(result, i, esampler) for do in things_to_do]
                hdf5.flush()
    if verbose:
        print('done production')

    return esampler, in_cent, in_prob


def emcee_burn(sampler, initial, nburn, model=None, prob0=None, verbose=True,
               **kwargs):
    """Run the emcee sampler for nburn iterations, reinitializing after each
    round.

    :param nburn:
        List giving the number of iterations in each round of burn-in.
        E.g. nburn=[32, 64] will run the sampler for 32 iterations before
        reinittializing and then run the sampler for another 64 iterations
    """
    limits = np.array(model.theta_bounds()).T
    if hasattr(model, 'theta_disp_floor'):
        disp_floor = model.theta_disp_floor()
    else:
        disp_floor = 0.0

    for k, iburn in enumerate(nburn[:-1]):
        epos, eprob, state = sampler.run_mcmc(initial, iburn, storechain=True)
        # find best walker position
        best = sampler.flatlnprobability.argmax()
        # is new position better than old position?
        if sampler.flatlnprobability[best] > prob0:
            prob0 = sampler.flatlnprobability[best]
            initial_center = sampler.flatchain[best, :]
        if epos.shape[0] < model.ndim*2:
            initial = reinitialize_ball(epos, eprob, center=initial_center,
                                        limits=limits, disp_floor=disp_floor,
                                        prior_check=model, **kwargs)
        else:
            initial = reinitialize_ball_covar(epos, eprob, center=initial_center,
                                              limits=limits, disp_floor=disp_floor,
                                              **kwargs)
        sampler.reset()
        if verbose:
            print('done burn #{}'.format(k))

    # Do the final burn-in
    epos, eprob, state = sampler.run_mcmc(initial, nburn[-1], storechain=False)
    if verbose:
        print('done all burn-in, starting production')
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
    pnew = resample_until_valid(multivariate_normal, center, Sigma, nwalkers,
                                **extras)
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

    pnew = resample_until_valid(sampler_ball, initial_center, scatter, nwalkers)
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


def pminimize(chi2fn, initial, args=None, model=None,
              method='powell', opts=None,
              pool=None, nthreads=1):
    """Do as many minimizations as you have threads, in parallel.  Always use
    initial_center for one of the minimization streams, the rest will be
    sampled from the prior for each parameter.  Returns each of the
    minimization result dictionaries, as well as the starting positions.
    """
    # Instantiate the minimizer
    mini = minimizer.Pminimize(chi2fn, args, opts,
                               method=method, pool=pool, nthreads=1)
    size = mini.size
    pinitial = minimizer_ball(initial, size, model)
    powell_guesses = mini.run(pinitial)

    return [powell_guesses, pinitial]


def reinitialize(best_guess, model, edge_trunc=0.1, reinit_params=[],
                 **extras):
    """Check if the Powell minimization found a minimum close to the edge of
    the prior for any parameter. If so, reinitialize to the center of the
    prior.

    This is only done for parameters where ``'reinit':True`` in the model
    configuration dictionary, or for parameters in the supplied
    ``reinit_params`` list.

    :param buest_guess:
        The result of some sort of optimization step, iterable of length
        model.ndim.

    :param model:
        A ..models.parameters.ProspectorParams() object.

    :param edge_trunc: (optional, default 0.1)
        The fractional distance from the edge of the priors that triggers
        reinitialization.

    :param reinit_params: optional
        A list of model parameter names to reinitialize, overrides the value or
        presence of the ``reinit`` key in the model configuration dictionary.

    :returns output:
        The best_guess with parameters near the edge reset to be at the center
        of the prior.  ndarray of shape (ndim,)
    """
    edge = edge_trunc
    bounds = model.theta_bounds()
    output = np.array(best_guess)
    reinit = np.zeros(model.ndim, dtype=bool)
    for p, inds in list(model.theta_index.items()):
        reinit[inds[0]:inds[1]] = (model._config_dict[p].get('reinit', False) or
                                   (p in reinit_params))

    for k, (guess, bound) in enumerate(zip(best_guess, bounds)):
        # Normalize the guess and the bounds
        prange = bound[1] - bound[0]
        g, b = guess/prange, bound/prange
        if ((g - b[0] < edge) or (b[1] - g < edge)) and (reinit[k]):
            output[k] = b[0] + prange/2
    return output


def minimizer_ball(center, nminimizers, model):
    """Setup a 'grid' of parameter values uniformly distributed between min and
    max More generally, this should sample from the prior for each parameter.
    """
    size = nminimizers
    pinitial = [center]
    if size > 1:
        ginitial = np.zeros([size - 1, model.ndim])
        for p, v in list(model.theta_index.items()):
            start, stop = v
            lo, hi = plotting_range(model._config_dict[p]['prior_args'])
            if model._config_dict[p]['N'] > 1:
                ginitial[:, start:stop] = np.array([np.random.uniform(l, h, size - 1)
                                                    for l, h in zip(lo, hi)]).T
            else:
                ginitial[:, start] = np.random.uniform(lo, hi, size - 1)
        pinitial += ginitial.tolist()
    return pinitial


def run_hmc_sampler(model, sps, lnprobf, initial_center, rp, pool=None):
    """Run a (single) HMC chain, performing initial steps to adjust the
    epsilon.
    """
    import hmc

    sampler = hmc.BasicHMC()
    eps = 0.
    # need to fix this:
    length = niter = nsegmax = nchains = None

    # initial conditions and calulate initial epsilons
    pos, prob, thiseps = sampler.sample(initial_center, model,
                                        iterations=10, epsilon=None,
                                        length=length, store_trajectories=False,
                                        nadapt=0)
    eps = thiseps
    # Adaptation of stepsize
    for k in range(nsegmax):
        # Advance each sampler after adjusting step size
        afrac = sampler.accepted.sum()*1.0/sampler.chain.shape[0]
        if afrac >= 0.9:
            shrink = 2.0
        elif afrac <= 0.6:
            shrink = 1/2.0
        else:
            shrink = 1.00

        eps *= shrink
        pos, prob, thiseps = sampler.sample(sampler.chain[-1, :], model,
                                            iterations=iterations,
                                            epsilon=eps, length=length,
                                            store_trajectories=False, nadapt=0)
        # this should not have actually changed during the sampling
        alleps[k] = thiseps
    # Main run
    afrac = sampler.accepted.sum()*1.0/sampler.chain.shape[0]
    if afrac < 0.6:
        eps = eps/1.5
    hpos, hprob, eps = sampler.sample(initial_center, model, iterations=niter,
                                      epsilon=eps, length=length,
                                      store_trajectories=False, nadapt=0)
    return sampler
