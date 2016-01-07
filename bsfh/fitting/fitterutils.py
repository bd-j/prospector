import sys
import numpy as np
import emcee
from . import minimizer
from ..models.priors import plotting_range

__all__ = ["run_emcee_sampler", "reinitialize_ball", "sampler_ball", "emcee_burn",
           "pminimize", "minimizer_ball", "reinitialize"]

def run_emcee_sampler(lnprobf, initial_center, model, verbose=True,
                      postargs=[], postkwargs={}, initial_prob=None,
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
        An instance of a models.ProspectrParams object.

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

    :param pool:
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
    disps = model.theta_disps(initial_center, initial_disp=initial_disp)
    initial = sampler_ball(initial_center, disps, nwalkers, model)

    # Initialize sampler
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf,
                                     args=postargs, kwargs=postkwargs,
                                     threads=nthreads, pool=pool)
    # Burn in sampler
    initial, in_cent, in_prob = emcee_burn(sampler, initial, nburn,
                                           initial_prob=initial_prob)
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
    for i, result in enumerate(esampler.sample(initial, iterations=niter, storechain=storechain)):
        if hdf5 is not None:
            chain[:, i, :] = result[0]
            lnpout[:, i] = result[1]
            if np.mod(i+1, int(interval*niter)) == 0:
                # do stuff every once in awhile
                # this would be the place to put some callback functions
                #[do(result, i, esampler) for do in things_to_do]
                hdf5.flush()
    if verbose:
        print('done production')

    return esampler, initial_center, initial_prob


def emcee_burn(sampler, initial, nburn, initial_prob=None):
    """Run the emcee sampler for nburn iterations, reinitializing after each
    round.

    :param nburn:
        List giving the number of iterations in each round of burn-in.
        E.g. nburn=[32, 64] will run the sampler for 32 iterations before
        reinittializing and then run the sampler for another 64 iterations
    """
    for k, iburn in enumerate(nburn[:-1]):
        epos, eprob, state = esampler.run_mcmc(initial, iburn, storechain=False)
        # find best walker position
        best = esampler.flatlnprobability.argmax()
        # is new position better than old position?
        if esampler.flatlnprobability[best] > initial_prob:
            initial_prob = esampler.flatlnprobability[best]
            initial_center = esampler.flatchain[best,:]
        initial = reinitialize_ball(initial_center, epos, nwalkers, model, **kwargs)
        esampler.reset()
        if verbose:
            print('done burn #{}'.format(k))

    # Do the final burn-in
    epos, eprob, state = esampler.run_mcmc(initial, nburn[-1], storechain=False)
    if verbose:
        print('done all burn-in, starting production')
    return epos, initial_center, inital_prob


def reinitialize_ball(initial_center, pos, nwalkers, model,
                      ptiles=[0.25, 0.5, 0.75], **extras):
    """Choose the best walker and build a ball around it based on the other
    walkers.
    """
    pos = np.atleast_2d(pos)
    tmp = np.percentile(pos, ptiles, axis=0)  
    # 1.35 is the ratio between the 25-75% interquartile range and 1
    # sigma (for a normal distribution)
    scatter = np.abs((tmp[2] -tmp[0]) / 1.35)
    if hasattr(model, 'theta_disp_floor'):
        disp_floor = model.theta_disp_floor(initial_center)
        scatter = np.sqrt(scatter**2 + disp_floor**2)    
    initial = sampler_ball(initial_center, scatter, nwalkers, model)
    return initial


def sampler_ball(center, disp, nwalkers, model):
    """Produce a ball around a given position, clipped to the prior range.
    """
    ndim = model.ndim
    initial = np.zeros([nwalkers, ndim])
    if np.size(disp) == 1:
        disp = np.zeros(ndim) + disp
    for p, v in list(model.theta_index.items()):
        start, stop = v
        lo, hi = plotting_range(model._config_dict[p]['prior_args'])
        try_param = (center[None, start:stop] +
                     (np.random.normal(0, 1, (nwalkers, stop-start)) *
                      disp[None, start:stop]))
        try_param = np.clip(try_param, np.atleast_1d(lo)[None, :],
                            np.atleast_1d(hi)[None, :])
        u = np.unique(try_param)
        if len(u) == 1:
            tweak = (np.random.uniform(0, 1, (nwalkers ,stop-start)) *
                     disp[None, start:stop])
            if u == lo:
                try_param += tweak
            elif u == hi:
                try_param -= tweak
                    
        initial[:, start:stop] = try_param
    return initial


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
    initial = sample_results['chain'][:,-1,:]
    nwalkers, ndim = initial.shape
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf, args = [model],
                                     threads = nthreads,  pool = pool)
    epos, eprob, state = esampler.run_mcmc(initial, niter, rstate0 =state)
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
                               method=method,
                               pool=pool, nthreads=1)
    size = mini.size
    pinitial = minimizer_ball(initial, size, model)
    powell_guesses = mini.run(pinitial)

    return [powell_guesses, pinitial]


def reinitialize(best_guess, model, edge_trunc=0.1,
                 reinit_params = [], **extras):
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
    reinit = np.zeros(model.ndim, dtype= bool)
    for p, inds in list(model.theta_index.items()):
        reinit[inds[0]:inds[1]] = (model._config_dict[p].get('reinit', False)
                                   or (p in reinit_params))
        
    for k, (guess, bound) in enumerate(zip(best_guess, bounds)):
        #normalize the guess and the bounds
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
        ginitial = np.zeros( [size -1, model.ndim] )
        for p, v in list(model.theta_index.items()):
            start, stop = v
            lo, hi = plotting_range(model._config_dict[p]['prior_args'])
            if model._config_dict[p]['N'] > 1:
                ginitial[:,start:stop] = np.array([np.random.uniform(l, h,size - 1)
                                                   for l,h in zip(lo,hi)]).T
            else:
                ginitial[:,start] = np.random.uniform(lo, hi, size - 1)
        pinitial += ginitial.tolist()
    return pinitial


def run_hmc_sampler(model, sps, lnprobf, initial_center, rp, pool=None):
    """Run a (single) HMC chain, performing initial steps to adjust the
    epsilon.
    """
    import hmc

    sampler = hmc.BasicHMC() 
    eps = 0.
    ##need to fix this:
    length = None
    niter = None
    nsegmax= None
    nchains = None
    
    #initial conditions and calulate initial epsilons
    pos, prob, thiseps = sampler.sample(initial_center, model,
                                        iterations = 10, epsilon = None,
                                        length = length, store_trajectories = False,
                                        nadapt = 0)
    eps = thiseps
    # Adaptation of stepsize
    for k in range(nsegmax):
        #advance each sampler after adjusting step size
        afrac = sampler.accepted.sum()*1.0/sampler.chain.shape[0]
        if afrac >= 0.9:
            shrink = 2.0
        elif afrac <= 0.6:
            shrink = 1/2.0
        else:
            shrink = 1.00
        
        eps *= shrink
        pos, prob, thiseps = sampler.sample(sampler.chain[-1,:], model,
                                            iterations = iterations,
                                            epsilon = eps, length = length,
                                            store_trajectories = False, nadapt = 0)
        alleps[k] = thiseps #this should not have actually changed during the sampling
    #main run
    afrac = sampler.accepted.sum()*1.0/sampler.chain.shape[0]
    if afrac < 0.6:
        eps = eps/1.5
    hpos, hprob, eps = sampler.sample(initial_center, model, iterations = niter,
                                      epsilon = eps, length = length,
                                      store_trajectories = False, nadapt = 0)
    return sampler
