import sys
import numpy as np
import emcee
from bsfh import minimizer
from bsfh.priors import plotting_range

try:
    import multiprocessing
except(ImportError):
    pass
    
def run_emcee_sampler(model, lnprobf, initial_center, rp, pool=None):
    """
    Run an emcee sampler, including iterations of burn-in and
    re-initialization.  Returns the production sampler.
    """
    # Parse input parameters
    ndim = rp['ndim']
    walker_factor = int(rp.get('walker_factor',4))
    nburn = rp['nburn']
    niter = int(rp['niter'])
    nthreads = int(rp.get('nthreads',1))
    initial_disp = rp['initial_disp']
    nwalkers = rp.get('nwalkers',
                      int(2 ** np.round(np.log2(ndim * walker_factor))))
    if rp['verbose']:
        print('number of walkers={}'.format(nwalkers))
    # Set up initial positions
    initial = np.zeros([nwalkers, ndim])
    for p, v in model.theta_index.iteritems():
        start, stop = v
        initial[:, start:stop] = (initial_center[start:stop] *
                                  np.random.normal(1, initial_disp, nwalkers)[:,None])
    # Initialize sampler
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf,
                                     threads = nthreads, args = [model], pool = pool)

    # Loop over the number of burn-in reintializations
    k=0
    for iburn in nburn[:-1]:
        epos, eprob, state = esampler.run_mcmc(initial, iburn)
        # Choose the best walker and build a ball around it based on
        #   the other walkers
        tmp = np.percentile(epos, [0.25, 0.5, 0.75], axis = 0)
        relative_scatter = np.abs((tmp[2] -tmp[0]) / tmp[1] / 1.35)
        best = np.argmax(eprob)
        initial = epos[best,:] * (1 + np.random.normal(0, 1, epos.shape) *
                                  relative_scatter[None,:]) 
        esampler.reset()
        k += 1
        if rp['verbose']:
            print('done burn #{}'.format(k))
    
    # Do the final burn-in
    epos, eprob, state = esampler.run_mcmc(initial, nburn[-1])
    initial = epos
    esampler.reset()
    if rp['verbose']:
        print('done all burn-in, starting production')

    # Production run
    epos, eprob, state = esampler.run_mcmc(initial, niter)
    if rp['verbose']:
        print('done production')

    return esampler

def restart_sampler(sample_results, lnprobf, sps, niter,
                    nthreads=1, pool=None):
    """
    Restart a sampler from its last position and run it for a
    specified number of iterations.  The sampler chain and the model
    object should be given in the sample_results dictionary.  Note
    that lnprobfn and sps must be defined at the global level in the
    same way as the sampler originally ran, or your results will be
    super weird (and wrong)!
    """
    model = sample_results['model']
    initial = sample_results['chain'][:,-1,:]
    nwalkers, ndim = initial.shape
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf, args = [model],
                                     threads = nthreads,  pool = pool)
    epos, eprob, state = esampler.run_mcmc(initial, niter, rstate0 =state)
    return esampler

        
def pminimize(function, model, initial_center, method='powell', opts=None,
              pool=None, nthreads=1):
    """
    Do as many minimizations as you have threads, in parallel.  Always
    use initial_center for one of the minimization streams, the rest
    will be sampled from the prior for each parameter.  Returns each
    of the minimization result dictionaries, as well as the starting
    positions.
    """
    
    # Instantiate the minimizer
    mini = minimizer.Pminimize(function, opts, model,
                               method=method,
                               pool=pool, nthreads=1)
    size = mini.size
    # Get initial positions to start minimizations
    pinitial = [initial_center]
    # Setup a 'grid' of parameter values uniformly distributed between
    #  min and max More generally, this should sample from the prior
    #  for each parameter
    if size > 1:
        ginitial = np.zeros( [size -1, model.ndim] )
        for p, v in model.theta_index.iteritems():
            start, stop = v
            hi, lo = plotting_range(model._config_dict[p]['prior_args'])
            if model._config_dict[p]['N'] > 1:
                ginitial[:,start:stop] = np.array([np.random.uniform(h, l,size - 1)
                                                   for h,l in zip(hi,lo)]).T
            else:
                ginitial[:,start] = np.random.uniform(hi, lo, size - 1)
        pinitial += ginitial.tolist()
    #print(mini.pool.size, mini.pool is None, len(pinitial))
    #sys.exit()
    #Actually run the minimizer
    powell_guesses = mini.run(pinitial)

    return [powell_guesses, pinitial]
    
def run_hmc_sampler(model, sps, lnprobf, initial_center, rp, pool=None):
    """
    Run a (single) HMC chain, performing initial steps to adjust the epsilon.
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
