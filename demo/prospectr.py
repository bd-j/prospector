#!/usr/local/bin/python

import time, sys, os
import numpy as np
import pickle

from bsfh import model_setup, write_results
from bsfh.gp import GaussianProcess
import bsfh.fitterutils as utils
from bsfh import model_setup

#########
# Read command line arguments
#########
argdict={'param_file':None, 'sptype':'sps_basis',
         'custom_filter_keys':None,
         'compute_vega_mags':False,
         'zcontinuous':True}
clargs = model_setup.parse_args(sys.argv, argdict=argdict)

#########
#SPS Model instance as global
########
sps = model_setup.load_sps(**clargs)

#GP instance as global
gp = GaussianProcess(None, None)

# Model as global
mod = model_setup.load_model(clargs['param_file'])

########
#LnP function as global
########
def lnprobfn(theta, obs):
    """
    Given a model object and a parameter vector, return the ln of the
    posterior.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param obs:
        a dictionary of observational data.

    :returns lnp:
        Ln posterior probability.
    """
    lnp_prior = mod.prior_product(theta)
    if np.isfinite(lnp_prior):
        
        # Generate mean model
        t1 = time.time()        
        mu, phot, x = mod.mean_model(theta, obs, sps = sps)
        d1 = time.time() - t1
        
        # Spectroscopy term
        t2 = time.time()
        if obs['spectrum'] is not None:
            mask = obs.get('mask', np.ones(len(obs['wavelength']),
                                           dtype= bool))
            gp.wave, gp.sigma = obs['wavelength'][mask], obs['unc'][mask]
            #use a residual in log space
            log_mu = np.log(mu)
            #polynomial in the log
            log_cal = (mod.calibration(theta, obs))
            delta = (obs['spectrum'] - log_mu - log_cal)[mask]
            gp.factor(mod.params['gp_jitter'], mod.params['gp_amplitude'],
                      mod.params['gp_length'], check_finite=False, force=False)
            lnp_spec = gp.lnlike(delta, check_finite=False)
        else:
            lnp_spec = 0.0

        # Photometry term
        if obs['maggies'] is not None:
            pmask = obs.get('phot_mask', np.ones(len(obs['maggies']),
                                                 dtype= bool))
            jitter = mod.params.get('phot_jitter',0)
            maggies = obs['maggies']
            phot_var = (obs['maggies_unc'] + jitter)**2
            lnp_phot =  -0.5*( (phot - maggies)**2 / phot_var )[pmask].sum()
            lnp_phot +=  -0.5*np.log(phot_var[pmask]).sum()
        else:
            lnp_phot = 0.0
        d2 = time.time() - t2
        
        if mod.verbose:
            print(theta)
            print('model calc = {0}s, lnlike calc = {1}'.format(d1,d2))
            fstring = 'lnp = {0}, lnp_spec = {1}, lnp_phot = {2}'
            values = [lnp_spec + lnp_phot + lnp_prior, lnp_spec, lnp_phot]
            print(fstring.format(*values))
        return lnp_prior + lnp_phot + lnp_spec
    else:
        return -np.infty
    
def chisqfn(theta, obs):
    return -lnprobfn(theta, obs)

#MPI pool.  This must be done *after* lnprob and
# chi2 are defined since slaves will only see up to
# sys.exit()
try:
    from emcee.utils import MPIPool
    pool = MPIPool(debug = False, loadbalance = True)
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
except(ValueError):
    pool = None
    
if __name__ == "__main__":

    ################
    # SETUP
    ################
    param_filename = clargs['param_file']
    rp = model_setup.run_params(param_filename)
    # Command line override of run_params
    rp = model_setup.parse_args(sys.argv, argdict=rp)
    rp['sys.argv'] = sys.argv
    if rp.get('mock', False):
        mock_info = model_setup.load_mock(param_filename, rp, mod, sps)
    else:
        obsdat = model_setup.load_obs(param_filename, rp)
    
    initial_theta = mod.initial_theta
    if rp['verbose']:
        print(mod.params)
    if rp.get('debug', False):
        try:
            pool.close()
        except:
            pass
        sys.exit()
        
    #################
    #INITIAL GUESS(ES) USING POWELL MINIMIZATION
    #################
    if rp['verbose']:
        print('minimizing chi-square...')
    ts = time.time()
    powell_opt = {'ftol': rp['ftol'], 'xtol':1e-6, 'maxfev':rp['maxfev']}
    powell_guesses, pinit = utils.pminimize(chisqfn, obsdat, initial_theta,
                                            method ='powell', opts=powell_opt,
                                            pool = pool, nthreads = rp.get('nthreads',1))
    
    best = np.argmin([p.fun for p in powell_guesses])
    best_guess = powell_guesses[best]
    pdur = time.time() - ts
    
    if rp['verbose']:
        print('done Powell in {0}s'.format(pdur))

    ###################
    #SAMPLE
    ####################
    #sys.exit()
    if rp['verbose']:
        print('emcee sampling...')
    tstart = time.time()
    initial_center = best_guess.x
    esampler = utils.run_emcee_sampler(lnprobfn, obsdat, initial_center,
                                       rp, pool = pool)
    edur = time.time() - tstart
    if rp['verbose']:
        print('done emcee in {0}s'.format(edur))

    ###################
    # PICKLE OUTPUT
    ###################
    write_results.write_pickles(rp, mod, esampler, powell_guesses,
                                toptimize=pdur, tsample=edur,
                                sampling_initial_center=initial_center)
    
    try:
        pool.close()
    except:
        pass
