import sys, os, getopt, subprocess
import numpy as np
import minimizer
import emcee
from readspec import *

try:
    import multiprocessing
except(ImportError):
    pass
    
lsun, pc = 3.846e33, 3.085677581467192e18 #in cgs
to_cgs = lsun/10**( np.log10(4.0*np.pi)+2*np.log10(pc*10) )

def run_command(cmd):
    """
    Open a child process, and return its exit status and stdout
    """
    child = subprocess.Popen(cmd, shell =True, stderr = subprocess.PIPE, 
                             stdin=subprocess.PIPE, stdout = subprocess.PIPE)
    out = [s for s in child.stdout]
    w = child.wait()
    return os.WEXITSTATUS(w), out

def parse_args(argv, rp = {'param_file':''}):
    
    shortopt = ''
    try:
        opts, args = getopt.getopt(argv[1:],shortopt,[k+'=' for k in rp.keys()])
    except getopt.GetoptError:
        print 'bsfh.py -- param_file <filename>'
        sys.exit(2)
    for o, a in opts:
        try:
            rp[o[2:]] = float(a)
        except:
            rp[o[2:]] = a
    if rp.get('verbose', False):
        print('reading parameters from {0}'.format(rp['param_file']))
    return rp

def run_emcee_sampler(model, sps, lnprobf, initial_center, rp, pool = None):
    """
    Run an emcee sampler, including iterations of burn-in and
    re-initialization.  Returns the production sampler.
    """
    # Parse input parameters
    ndim = rp['ndim']
    walker_factor = int(rp['walker_factor'])
    nburn = rp['nburn']
    niter = int(rp['niter'])
    nthreads = int(rp['nthreads'])
    initial_disp = rp['initial_disp']
    nwalkers = int(2 ** np.round(np.log2(ndim * walker_factor)))

    # Set up initial positions
    initial = np.zeros([nwalkers, ndim])
    for p, d in model.theta_desc.iteritems():
        start, stop = d['i0'], d['i0']+d['N']
        hi, lo = d['prior_args']['maxi'], d['prior_args']['mini']
        initial[:, start:stop] = (np.random.normal(1, initial_disp, nwalkers)[:,None] *
                                  initial_center[start:stop])
    # Initialize sampler
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf,
                                     threads = nthreads, args = [model], pool = pool)

    # Loop over the number of burn-in reintializitions
    for iburn in nburn[:-1]:
        epos, eprob, state = esampler.run_mcmc(initial, iburn)
        # Choose the best walker and build a ball around it based on
        #   the other walkers
        tmp = np.percentile(epos, [0.25, 0.5, 0.75], axis = 0)
        relative_scatter = np.abs(1.5 * (tmp[2] -tmp[0])/tmp[1])
        best = np.argmax(eprob)
        initial = epos[best,:] * (1 + np.random.normal(0, 1, epos.shape) * relative_scatter[None,:]) 
        esampler.reset()

    # Do the final burn-in
    epos, eprob, state = esampler.run_mcmc(initial, nburn[-1])
    initial = epos
    esampler.reset()

    # Production run
    epos, eprob, state = esampler.run_mcmc(initial, niter)

    return esampler

def restart_sampler(sample_results, lnprobf, sps, niter,
                    nthreads = 1, pool = None):
    """
    Restart a sampler from its last position and run it for a
    specified number of iterations.  The sampler chain should be given
    in the sample_results dictionary.  Note that lnprobfn and sps must
    be defined at the global level in the same way as the sampler
    originally ran.
    """
    model = sample_results['model']
    initial = sample_results['chain'][:,-1,:]
    nwalkers, ndim = initial.shape
    esampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobf, args = [model],
                                     threads = nthreads,  pool = pool)
    epos, eprob, state = esampler.run_mcmc(initial, niter, rstate0 =state)
    pass

        
def pminimize(function, model, initial_center, method ='powell', opts = None,
              pool = None, nthreads = 1):
    """
    Do as many minimizations as you have threads, in parallel.  Always
    use initial_center for one of the minimization streams, the rest
    will be sampled from the prior for each parameter.  Returns each
    of the minimization result dictionaries, as well as the starting
    positions.
    """
    
    # Instantiate the minimizer
    mini = minimizer.Pminimize(function, method, opts, model,
                               pool = pool, nthreads = 1)
    size = mini.size
    # Get initial positions to start minimizations
    pinitial = [initial_center]
    # Setup a 'grid' of parameter values uniformly distributed between
    #  min and max More generally, this should sample from the prior
    #  for each parameter
    if size > 1:
        ginitial = np.zeros( [size -1, model.ndim] )
        for p, d in model.theta_desc.iteritems():
            start, stop = d['i0'], d['i0']+d['N']
            hi, lo = d['prior_args']['maxi'], d['prior_args']['mini']
            if d['N'] > 1:
                ginitial[:,start:stop] = np.array([np.random.uniform(h, l,size - 1)
                                                   for h,l in zip(hi,lo)]).T
            else:
                ginitial[:,start] = np.random.uniform(hi, lo, size - 1)
        pinitial += ginitial.tolist()
    print(mini.pool.size, mini.pool is None, len(pinitial))
    #sys.exit()
    #Actually run the minimizer
    powell_guesses = mini.run(pinitial)

    return [powell_guesses, pinitial]
    
