#script for playing with SFH inference at fixed population parameters

import time, sys
import pickle
import numpy as np
from scipy.optimize import minimize

import emcee, priors

import observate, attenuation
import sps_basis, sedmodel

snr = 50.  #Signal/Noise ratio of the mock
nage = 10  #number of ages to model
logt0 = 7.0
dlogt = (np.log10(13.7e9) - 7.0) / (nage-1)
filters = ['sdss_r0']
if filters: filters = observate.load_filters(filters)

###############
# SET UP MODEL
###############
sps = sps_basis.StellarPopBasis()

ages = 10**(np.arange(nage)*dlogt + logt0)/1e9
metals = np.array([0])
ncomp = len(metals) * len(ages)

masstheta = {'i0':0, 'N': ncomp, 'lower':0., 'dtype':'<f8', 
             'prior_function':priors.positive,
             'prior_gradient_function':priors.zeros,
             'prior_args':{}}

dusttheta = {'i0':ncomp, 'N': 1, 'lower':0., 'upper': 2, 'dtype':'<f8', 
             'prior_function':priors.zeros,
             'prior_gradient_function':priors.zeros,
             'prior_args':{}}

theta_desc = {'mass':masstheta}  #, 'dust2':dusttheta}
fixed_params = {'sfh':0, 'vel_broad':20, 'imf_type': 2, 'imf3': 2.3,
                'dust_type': 1, 'dust1':0., 'dust2':0., 'dust_curve':attenuation.calzetti}

model = sedmodel.SedModel(theta_desc, tage = ages, zmet = metals, **fixed_params)
#model.sps = sps
def lnprobfn(theta, mod):
    """wrapper on the model instance method, defined here
    globally to enable multiprocessing"""
    return mod.lnprob(theta, sps = sps)


def calibration():
    return 1.0 #simple calibration model
model.calibration = calibration

###############
# MOCK OBSERVATIONS
################
obs = {}
obs['wavelength'] = sps.ssp.wavelengths[10:-10]
obs['filters'] = filters
model.obs = obs
mock_mass =  np.log(model.params['tage']/model.params['tage'].min()) + 0.1
mock_mass = mock_mass[:,None] * (2 + model.params['zmet'][None,:])
mock_theta = mock_mass.flatten()
mock_spec, mock_phot, mock_other = model.model(mock_theta, sps =sps)

obs['spectrum'] = mock_spec * (1 + np.random.normal(0,1./snr, len(obs['wavelength'])))
obs['unc'] = mock_spec/snr
obs['mask'] = ((obs['wavelength'] > 2e3) & (obs['wavelength'] < 20e3))
if obs['filters'] is not None:
    obs['mags'] = -2.5 * np.log10(mock_phot)
    obs['mags_unc'] = np.zeros_like(obs['mags'])+0.05
else:
    obs['filters'] = None

model.add_obs(obs)
model.ndof = len(model.obs['wavelength']) + len(model.obs['mags'])
model.verbose = False

###############
# INITIALIZE WITH BFGS
################
def chi2(theta):
    """A sort of chi2 function that allows for maximization of lnP"""
    return -model.lnprob(theta, sps =sps)

def chi2_grad(theta):
    """A sort of chi2 gradient function that allows for maximization of lnP"""
    return -model.lnprob_grad(theta, sps=sps)

bounds = len(mock_theta) * [(0, None)]
bfgs_opt = {'ftol':1e-14, 'gtol':1e-12}
x0 = mock_theta * np.clip(np.random.normal(5.0,5.0, len(mock_theta)),0,np.infty)
bfgs_guess = minimize(chi2, x0, method='L-BFGS-B', jac = chi2_grad,
                      options=bfgs_opt, bounds = bounds)

###############
#SAMPLE WITH HMC
###############

ndim = len(mock_theta)
nsampler = 1
nsegmax = 10
iterations = 50
length = 10

nwalkers = 256
niterations = 2e4
nthreads =1

tstart = time.time()
hinitial =  bfgs_guess.x.copy()
initial_disp = 0.01

initial = np.zeros([nwalkers, ndim])
for p, d in model.theta_desc.iteritems():
    start, stop = d['i0'], d['i0']+d['N']
    initial[:, start:stop] = np.random.normal(1, initial_disp, nwalkers)[:,None] * hinitial[start:stop]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn, threads = nthreads, args = [model])
epos, eprob, state = sampler.run_mcmc(initial, niterations)


# # of likelihood calls
#hmc burn in 10 * 50 * 10
#hmc production 800 * 5000

#emcee burn-in 256 * 25
#emcee production 256 * 20000


dur = time.time() - tstart

#######
# SAVE RESULTS
#######

results = {}
results['mock_input_theta'] = mock_theta
results['obs'] = model.obs
results['theta'] = model.theta_desc
results['initial_center'] = hinitial
results['chain'] = sampler.chain
results['lnprobability'] = sampler.lnprobability
results['acceptance'] = sampler.acceptance_fraction
results['duration'] = dur
model.sps = None
results['model'] = model
    
out = open('sfhdemo_t{0}_emcee.p'.format(int(time.time())), 'wb')
pickle.dump(results, out)
out.close()


