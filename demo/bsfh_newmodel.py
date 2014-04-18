#script for playing with SFH inference at fixed population parameters


import time
import sys
import numpy as np
from scipy.optimize import minimize

import observate

import sps_basis, newmodel, priors
import newmodel

import hmc, emcee, acor
from convergence import gelman_rubin_R as grR

rp = {}
snr = 50.
nage = 20
logt0 = 7.0
dlogt = (np.log10(13.7e9) - 7.0) / (nage-1)
filters = ['sdss_r0']
if filters: filters = observate.load_filters(filters)

###############
# SET UP
###############
sps = sps_basis.StellarPopBasis()

ages = 10**(np.arange(nage)*dlogt + logt0)/1e9
metals = np.array([0])
ncomp = len(metals) * len(ages)

masstheta = {'i0':0, 'N': ncomp, 'lower':0., 'dtype':'<f8', 
             'prior_function':priors.zeros,
             'prior_gradient_function':priors.zeros,
             'prior_args':{}}

dusttheta = {'i0':ncomp, 'N': 1, 'lower':0., 'upper': 2, 'dtype':'<f8', 
             'prior_function':priors.zeros,
             'prior_gradient_function':priors.zeros,
             'prior_args':{}}

theta_desc = {'mass':masstheta}  #, 'dust2':dusttheta}
model = newmodel.CompositeModel(theta_desc, ages, metals, sps =sps)
ndim = 0
for p, d in model.theta_desc.iteritems():
    ndim += d['N']
rp['ndim'] = ndim

sps_fixed_params = {'sfh':0, 'imf_type': 2, 'dust_type': 1, 'imf3': 2.3, 'dust2':0., 'dust1':0., 'vel_broad':20}
for k, v in sps_fixed_params.iteritems():
    model.params[k] = v

model.sps = sps

###############
# MOCK
################
obs = {}
obs['wavelength'] = sps.sps.wavelengths
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
# INITIALIZE
################
def chi2(theta):
    """A sort of chi2 function that allows for maximization of lnP"""
    return -model.lnprob(theta, sps =sps)

def chi2_grad(theta):
    """A sort of chi2 gradient function that allows for maximization of lnP"""
    return -model.lnprob_grad(theta, sps=sps)

bounds = len(mock_theta) * [(0, None)]
bfgs_opt = {'ftol':1e-14, 'gtol':1e-12, 'maxfev':1e4}
x0 = mock_theta * np.clip(np.random.normal(5.0,5.0, len(mock_theta)),0,np.infty)
bfgs_guess = minimize(chi2, x0, method='L-BFGS-B', jac = chi2_grad,
                      options=bfgs_opt, bounds = bounds)

###############
#SAMPLE
###############

ndim = len(mock_theta)
nsampler = 1
nsegmax = 20
iterations = 50
length = 50

maxshrink = 2.0
target = 0.8
alleps = np.zeros(nsegmax)

#hmc
tstart = time.time()
#hinitial = np.random.uniform(0,2 * tweight.mean(),ndim)
#hinitial[-1] = np.random.uniform(0,2)
sampler = hmc.BasicHMC() 
eps = 0.
#initial conditions and calulate initial epsilons
hinitial =  bfgs_guess.x.copy()
pos, prob, thiseps = sampler.sample(hinitial, model, iterations = 10, epsilon = None,
                                    length = length, store_trajectories = False, nadapt = 0)
eps = thiseps
   

print('burn-in')
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
    print('shrink = {0}'.format(shrink))
    pos, prob, thiseps = sampler.sample(sampler.chain[-1,:], model, iterations = iterations,
                                        epsilon = eps, length = length, store_trajectories = False, nadapt = 0)
    alleps[k] = thiseps #this should not have actually changed during the sampling
 

        
# Production run
afrac = sampler.accepted.sum()*1.0/sampler.chain.shape[0]
if afrac < 0.5:
    eps = eps/2.0

hpos, hprob, eps = sampler.sample(hinitial, model, iterations = 10000,
                                   epsilon = eps, length = length*10,
                                   store_trajectories = False, nadapt = 0)


dur = time.time() - tstart

#try:
#    tau, mean, sigma = acor.acor(hsampler.lnprob)
#except:
#    pass



