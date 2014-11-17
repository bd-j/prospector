#!/usr/local/bin/python

import time, sys, os
import numpy as np
import pickle

from bsfh import model_setup, write_results
from bsfh.gp import GaussianProcess
import bsfh.fitterutils as utils
from bsfh import model_setup

sptype = model_setup.parse_args(sys.argv).get('sps_type', 'sps_basis')

#SPS Model as global
if sptype == 'sps_basis':
    from bsfh import sps_basis
    sps = sps_basis.StellarPopBasis()
elif sptype == 'fsps':
    import fsps
    sps = fsps.StellarPopulation()

#GP instance as global
gp = GaussianProcess(None, None)

#LnP function as global
def lnprobfn(theta, mod):
    """
    Given a model object and a parameter vector, return the ln of the
    posterior.
    """
    lnp_prior = mod.prior_product(theta)
    if np.isfinite(lnp_prior):
        
        # Generate mean model
        t1 = time.time()        
        mu, phot, x = mod.mean_model(theta, sps = sps)
        d1 = time.time() - t1
        
        # Spectroscopy term
        t2 = time.time()
        if mod.obs['spectrum'] is not None:
            mask = mod.obs['mask']
            gp.wave, gp.sigma = mod.obs['spectrum'][mask], mod.obs['unc'][mask]
            #use a residual in log space
            log_mu = np.log(mu)
            #polynomial in the log
            log_cal = (mod.calibration(theta))
            delta = (mod.obs['spectrum'] - log_mu - log_cal)[mask]
            gp.factor(mod.params['gp_jitter'], mod.params['gp_amplitude'],
                      mod.params['gp_length'], check_finite=False, force=False)
            lnp_spec = gp.lnlike(delta)
        else:
            lnp_spec = 0.0

        # Photometry term
        if mod.obs['mags'] is not None:
            pmask = mod.obs.get('phot_mask',
                                np.ones(len(mod.obs['mags']), dtype= bool))
            jitter = mod.params.get('phot_jitter',0)
            maggies = 10**(-0.4 * mod.obs['mags'])
            phot_var = maggies**2 * ((mod.obs['mags_unc']**2 + jitter**2)/1.086**2)
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
    
def chisqfn(theta, mod):
    return -lnprobfn(theta, mod)

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
    inpar = model_setup.parse_args(sys.argv)
    parset, model = model_setup.setup_model(inpar['param_file'], sps=sps)
    parset.run_params['ndim'] = model.ndim
    _ = model_setup.parse_args(sys.argv, argdict=parset.run_params)
    parset.run_params['sys.argv'] = sys.argv
    rp = parset.run_params #shortname
    initial_theta = parset.initial_theta
    
    #################
    #INITIAL GUESS(ES) USING POWELL MINIMIZATION
    #################
    if rp['verbose']:
        print('Minimizing')
    ts = time.time()
    powell_opt = {'ftol': rp['ftol'], 'xtol':1e-6, 'maxfev':rp['maxfev']}
    powell_guesses, pinit = utils.pminimize(chisqfn, model, initial_theta,
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
        print('emcee...')
    tstart = time.time()
    initial_center = best_guess.x
    esampler = utils.run_emcee_sampler(model, lnprobfn, initial_center, rp, pool = pool)
    edur = time.time() - tstart
    if rp['verbose']:
        print('done emcee in {0}s'.format(edur))

    ###################
    # PICKLE OUTPUT
    ###################
    write_results.write_pickles(parset, model, esampler, powell_guesses,
                                toptimize=pdur, tsample=edur,
                                sampling_initial_center=initial_center)
    
    try:
        pool.close()
    except:
        pass
