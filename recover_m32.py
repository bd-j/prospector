import os, time
import numpy as np

import fsps

import fitsfh
import plotter as pl
import utils


#ideally these are read from command line.....
rp = {'snr': 100.,
      'name': 'M32',
      'veldisp': 0., # in km/s
      'wlo':1.3e3, 'whi':2e4, #in angstroms
      'spstype':'MILES+Padova'}

rp['figname'] = "hires_{0}".format(rp['name'])
rp['nbin'] = nbin
rp['met_index'] = 4

rp['nburn'] = 300
rp['nthreads'] = 1
rp['nsteps'] = 600
rp['walker_factor'] = 40

####### SET UP THE TIME BINS OF WHICH YOU'LL COMBINE THE SPECTRA ###########
#note that both the CSP spectra and the SFHs to recover should
#have the same time binning.
isoc_ages = 10**np.arange(6.6,10.1,0.05)

dtype = {'names':('start', 'end', 'center'),
       'formats':('<f8','<f8','<f8')}
age_bins = np.zeros(nbin, dtype = dtype)

dlogt = 0.25
logt0 = 6.5
nbin = 14

age_bins['start'] = 10**( np.arange(nbin) * dlogt + logt0 ) #in yrs
age_bins['end'] = 10**( np.arange(nbin) * dlogt +logt0 + dlogt )   #in yrs
age_bins['start'][0] = 0.0
age_bins['end'][-1] = 13.7e9
age_bins['center'] = (age_bins['start']+age_bins['end'])/2.


###### GENERATE THE BASIS SPECTRA ######
sps = fsps.StellarPopulation(imf_type = 2, dust_type = 0, dust1 = 0.0, dust2 = 0.0)
fitter = fitsfh.FixedZ(rp, sps) #single metallicity

fitter.load_spectra(age_bins)

###### INPUT SFH ##
fitter.mock_sfh = utils.m32_sfh(age_bins)

###### SETUP OUTPUT ########
waves_lo = np.arange(3000, 4000, 200)
waves_hi = np.arange(6000, 11000, 1000)
snr_values = np.array([20,60,100,140])
sigma_sfr = np.zeros([len(waves_lo), len(waves_hi), len(snr_values), len(age_bins)+1])

###LOOP OVER FIT PARAMETERS
for i,wlo in enumerate(waves_lo):
	fitter.rp['wlo'] = wlo
	for j,whi in enumerate(waves_hi):
		fitter.rp['whi'] = whi
		for k, snr in enumerate(snr_values):
			fitter.rp['snr'] = snr
			#Mock the spectrum and its error array
			fitter.mock_spectrum()
			#run the mcmc smapler
			fitter.fit_sfh(fitter.rp)
			#collect scalar variables summarizing the fit quality
			sigma_sfr[i,j,k,:] = fitter.recovery_stats()


##### PLOTS ########
