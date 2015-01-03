#!/usr/local/bin/python

import time, sys, os
import numpy as np
np.errstate(invalid='ignore')
import pickle

from bsfh import model_setup, write_results
import bsfh.fitterutils as utils
from bsfh.likelihood import LikelihoodFunction

#########
# Read command line arguments
#########
sargv = sys.argv
argdict={'param_file':None, 'sptype':'sps_basis',
         'custom_filter_keys':None,
         'compute_vega_mags':False,
         'zcontinuous':1,
         'gptype': ''}
clargs = model_setup.parse_args(sargv, argdict=argdict)
run_params = model_setup.get_run_params(argv = sargv, **clargs)

#########
# Globals
########
# SPS Model instance as global
sps = model_setup.load_sps(**clargs)
# GP instance as global
gp_spec = model_setup.load_gp(**clargs)
# Model as global
global_model = model_setup.load_model(param_file=clargs['param_file'])
# Obs as global
global_obs = model_setup.load_obs(**run_params)

########
#LnP function as global
########
likefn = LikelihoodFunction(obs=global_obs, model=global_model)

# the simple but obscuring way.  Difficult for users to change
def obscured_lnprobfn(theta, model = None, obs = None):
    if model is None:
        model = global_model
    if obs is None:
        obs = global_obs
    return likefn.lnpostfn(theta, model=model, obs=obs,
                           sps=sps, gp=gp_spec)

# the more explicit way
def lnprobfn(theta, model=None, obs=None, verbose=run_params['verbose']):
    """ Given a parameter vector and optionally a dictionary of
    observational ata and a model object, return the ln of the
    posterior. This requires that an sps object (and if using spectra
    and gaussian processes, a GP object) be instantiated.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param model:
        bsfh.sedmodel model object, with attributes including
        ``params``, a dictionary of model parameters.  It must also have
        ``prior_product()``, and ``mean_model()`` methods
        defined.

    :param obs:
        A dictionary of observational data.  The keys should be
          *``wavelength``
          *``spectrum``
          *``unc``
          *``maggies``
          *``maggies_unc``
          *``filters``
          * and optional spectroscopic ``mask`` and ``phot_mask``.
        
    :returns lnp:
        Ln posterior probability.
    """
    if model is None:
        model = global_model
    if obs is None:
        obs = global_obs
        
    lnp_prior = model.prior_product(theta)
    if np.isfinite(lnp_prior):
        # Generate mean model and GP kernel(s)
        t1 = time.time()        
        mu, phot, x = model.mean_model(theta, obs, sps = sps)
        try:
            s, a, l = model.spec_gp_params()
            gp_spec.kernel[:] = np.log(np.array([s[0],a[0]**2,l[0]**2]))
        except(AttributeError):
            #There was no spec_gp_params method
            pass
        d1 = time.time() - t1

        #calculate likelihoods
        t2 = time.time()
        lnp_spec = likefn.lnlike_spec(mu, obs=obs, gp=gp_spec)
        lnp_phot = likefn.lnlike_phot(phot, obs=obs, gp=None)
        d2 = time.time() - t2
        if verbose:
            write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

        return lnp_prior + lnp_phot + lnp_spec
    else:
        return -np.infty
    
def chisqfn(theta, model, obs):
    """Negative of lnprobfn for minimization, and also handles passing
    in keyword arguments which can only be postional arguments when
    using scipy minimize.
    """
    return -lnprobfn(theta, model=model, obs=obs)

def write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2):
    """Write all sorts of documentary info for debugging.
    """
    print(theta)
    print('model calc = {0}s, lnlike calc = {1}'.format(d1,d2))
    fstring = 'lnp = {0}, lnp_spec = {1}, lnp_phot = {2}'
    values = [lnp_spec + lnp_phot + lnp_prior, lnp_spec, lnp_phot]
    print(fstring.format(*values))

##################
# MPI pool.  This must be done *after* lnprob and
# chi2 are defined since slaves will only see up to
# sys.exit()
##################
try:
    from emcee.utils import MPIPool
    pool = MPIPool(debug = False, loadbalance = True)
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
except(ValueError):
    pool = None
    print('Not using MPI')

def halt():
    """Exit, closing pool safely.
    """
    print('stopping for debug')
    try:
        pool.close()
    except:
        pass
    sys.exit(0)


############
# Master branch
#############
    
if __name__ == "__main__":

    ################
    # Setup
    ################
    rp = run_params
    rp['sys.argv'] = sys.argv
    # Reload model and obs from specific files?
    #model = model_setup.load_model(param_file=clargs['param_file'])
    #obsdat = model_setup.load_obs(**rp)
    #chi2args = [model, obsdat]
    #postkwargs = {'obs':obsdat, 'model':model}
    # Or just use the globals?
    model = global_model
    obsdat = global_obs
    chi2args = [None, None]
    postkwargs = {}
    
    initial_theta = model.initial_theta
    if rp.get('debug', False):
        halt()
        
    #################
    # Initial guesses using powell minimization
    #################
    if bool(rp.get('do_powell', True)):
        if rp['verbose']:
            print('minimizing chi-square...')
        ts = time.time()
        powell_opt = {'ftol': rp['ftol'], 'xtol':1e-6, 'maxfev':rp['maxfev']}
        powell_guesses, pinit = utils.pminimize(chisqfn, initial_theta,
                                                args=chi2args, model=model,
                                                method ='powell', opts=powell_opt,
                                                pool = pool, nthreads = rp.get('nthreads',1))
        best = np.argmin([p.fun for p in powell_guesses])
        initial_center = powell_guesses[best].x
        pdur = time.time() - ts
        if rp['verbose']:
            print('done Powell in {0}s'.format(pdur))
    else:
        powell_guesses = None
        pdur = 0.0
        initial_center = initial_theta.copy()
        
    ###################
    # Sample
    ####################
    if rp['verbose']:
        print('emcee sampling...')
    tstart = time.time()
    esampler = utils.run_emcee_sampler(lnprobfn, initial_center, model,
                                       postkwargs=postkwargs, pool = pool, **rp)
    edur = time.time() - tstart
    if rp['verbose']:
        print('done emcee in {0}s'.format(edur))

    ###################
    # Pickle Output
    ###################
    write_results.write_pickles(rp, model, obsdat,
                                esampler, powell_guesses,
                                toptimize=pdur, tsample=edur,
                                sampling_initial_center=initial_center)
    
    try:
        pool.close()
    except:
        pass
