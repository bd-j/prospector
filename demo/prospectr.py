#!/usr/local/bin/python

import time, sys, os
import numpy as np
import pickle

from bsfh import model_setup, sps_basis
from bsfh.gp import GaussianProcess
import bsfh.fitterutils as utils
from bsfh import model_setup

#SPS Model as global
sps = sps_basis.StellarPopBasis()

#Gp instance as global
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

    #############
    # SETUP
    ################
    inpar = model_setup.parse_args(sys.argv)
    model, parset = model_setup.setup_model(inpar['param_file'])
    parset.run_params['ndim'] = model.ndim
    rp = parset.run_params
    initial_center = parsert.initial_theta
    
    #################
    #INITIAL GUESS USING POWELL MINIMIZATION
    #################
    if rp['verbose']:
        print('Minimizing')
    ts = time.time()
    powell_opt = {'ftol': rp['ftol'], 'xtol':1e-6,
                'maxfev':rp['maxfev']}
    powell_guesses, pinit = utils.pminimize(chisqfn, model, initial_center,
                                       method ='powell', opts=powell_opt,
                                       pool = pool, nthreads = rp['nthreads'])
    
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
    
    #nsamplers = int(rp['nsamplers'])
    theta_init = initial_center
    initial_center = best_guess.x #np.array([8e3, 2e-2, 0.5, 0.1, 0.1, norm])
    esampler = utils.run_emcee_sampler(model, lnprobfn, initial_center, rp, pool = pool)
    edur = time.time() - tstart

    ###################
    # PICKLE OUTPUT
    ###################
    results, model_store = {}, {}
    
    results['run_params'] = rp
    results['obs'] = model.obs
    results['plist'] = parset.model_params
    results['pardict'] =  modeldef.plist_to_pdict([modeldef.functions_to_names(p)
                                                  for p in parset.model_params])
    results['initial_center'] = initial_center
    results['initial_theta'] = theta_init
    
    results['chain'] = esampler.chain
    results['lnprobability'] = esampler.lnprobability
    results['acceptance'] = esampler.acceptance_fraction
    results['duration'] = edur
    results['optimizer_duration'] = pdur

    model_store['powell'] = powell_guesses
    model_store['model'] = model
    #pull out the git hash for bsfh here.
    bsfh_dir = os.path.dirname(modeldef.__file__)
    bgh = utils.run_command('cd ' + bsfh_dir +
                           '\n git rev-parse HEAD')[1][0].replace('\n','')
    cgh = utils.run_command('git rev-parse HEAD')[1][0].replace('\n','')

    results['bsfh_version'] = bgh
    results['cetus_version'] = cgh
    model_store['bsfh_version'] = bgh
    
    tt = int(time.time())
    out = open('{1}_{0}_mcmc'.format(tt, rp['outfile']), 'wb')
    pickle.dump(results, out)
    out.close()

    out = open('{1}_{0}_model'.format(tt, rp['outfile']), 'wb')
    pickle.dump(model_store, out)
    out.close()
    
    try:
        pool.close()
    except:
        pass
