import os, time
import numpy as np

import emcee
import fsps
import plotter as pl
import utils
#import sfhs
#import observate
#import likelihood

#ideally these are read from command line.....
pars = {'snr': 50.,
        'name': 'ddo190',
        'veldisp': 0., # in km/s
        'wlo':1.3e3, 'whi':2e4, #in angstroms
        'spstype':'BaSel+Padova'}
pars['figname'] = "hires_{0}".format(pars['name'])

angst_dat = utils.load_angst_sfh(pars['name'])

####### SET UP THE TIME BINS OF WHICH YOU'LL COMBINE THE SPECTRA ###########

#note that both the CSP spectra and the SFHs to recover should
#have the same time binning.
bin_starts = 10**np.arange(6.6,10.1,0.1) #in yrs
bin_ends = 10**np.arange(6.5,10.0,0.1)   #in yrs
bin_ends[0] = 0.0
fitsfilename=['data/models/fspsSalp_angstBin%s_z2.fits' % str(s) for s in range(36)]
bin_centers = (bin_starts+bin_ends)/2.

#parameters to recombine into wider bins.  In principle this should not be required
subs = np.array([0,0,0,
                 1,1,1,1,1,
                 2,2,2,2,2,
                 3,3,3,3,3,3,3,
                 4,4,4,4,4,
                 5,5,5,5,5,
                 6,6,6,6,6
    ])
nrebin = subs.max()+1
pars['nbin'] = nrebin

pars['rebin_starts'] = bin_starts[np.searchsorted(subs,np.arange(nrebin), side = 'right') -1]
pars['rebin_ends'] = bin_ends[np.searchsorted(subs,np.arange(nrebin), side = 'left')]
pars['rebin_centers'] = (pars['rebin_starts']+pars['rebin_ends'])/2./1e9 #in Gyr

###### GENERATE THE BASIS SPECTRA ######

sps = fsps.StellarPopulation(imf_type = 2, dust_type = 0, dust1 = 0.0, dust2 = 0.0)
model_wave = sps.wavelengths
spec_array = np.zeros( [nrebin,sps.wavelengths.shape[0]] )
mstar_array = np.zeros( nrebin )
angst_sfh = np.zeros( nrebin )

print

for i in range(bin_starts.shape[0]):
    ## spectra are normalized to an SFR of one for each component
    wave, spec = sps.get_spectrum(zmet = 4, tage = bin_centers[i]*1e-9, peraa=True) 
    spec_array[subs[i],:] += (spec/10 * (bin_starts[i]-bin_ends[i]))#cumulate into the wider bins with sfr normalization
    mstar_array[subs[i]] += 10**sps.log_mass/10

    angst_sfh[subs[i]] += angst_dat['sfr'][i]*(bin_starts[i]-bin_ends[i])

###### LOAD OR MOCK THE OBSERVATIONS######
angst_sfh = angst_sfh / (pars['rebin_starts']-pars['rebin_ends'])
mask = np.where((sps.wavelengths > pars['wlo']) & (sps.wavelengths < pars['whi']), 1, 0) 
mock_spec = (angst_sfh * spec_array.T).sum(axis = 1)
noised_mock = mock_spec * (1+np.random.normal(0,1,sps.wavelengths.shape[0])/pars['snr'])
data = noised_mock
err = mock_spec/pars['snr']

#####probability function#####

def lnprob(theta, data, err, mask):
    #prior bounds test.  all components must be non-negative
    #prior probabilities could also be introduced here via a priors blob in the call?
    ptest=[]
    for i,par in enumerate(theta):
        ptest.append(par >= 0)

    if (False in ptest):
        #set lnp to -infty if some component is negative
        return -np.infty
    else:  
        sfrs = theta #[:nbin] #not necessary
        model_flux = (sfrs * spec_array.T).sum(axis = 1)
        delta = (model_flux - data)/err*mask
        return -0.5*np.sum(delta**2)

#####define parameter ranges#####

ndim = nrebin
nwalkers = ndim * 10
#make a guess based on the average sfr.  in practice this can be worked out pretty well a priori from real data even
initial = np.random.uniform(size = (nwalkers,ndim)) * angst_sfh.mean() * 5
#MCMC chain 
nburn = 300
nsteps = 700
nthreads = 1 #to allow easy killing

#burn in the chain, reset, and run
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data,err,mask])
tstart = time.time()
pos,prob,state = sampler.run_mcmc(initial, nburn)
sampler.reset()
sampler.run_mcmc(np.array(pos),nsteps, rstate0=state)

duration = time.time()-tstart
print('Sampling done in', duration, 's')
print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))

outsamples = sampler.flatchain
#outprob = sampler.flatlnprobability

pars['nsample'] = outsamples.shape[0]
pars['maf'] = np.mean(sampler.acceptance_fraction)
pars['tsample'] = duration

##### PLOTS ########
pl.plot_sfh(pars, outsamples, angst_sfh)

pl.plot_spectrum(pars, outsamples, angst_sfh, model_wave, spec_array)

pl.plot_sfr_mass(pars, outsamples, angst_sfh, sfr_time_index = 3)

pl.plot_covariance(pars, outsamples, bin1 = 0, bin2 = 1, sfh_input = angst_sfh)
