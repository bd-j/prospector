# Module containing classes to generate spectra of composite populations,
# given an SSP object and an SFH object
# Alternatively, spectra may be read (and written) to fits binary tables
import numpy as np
import pyfits
from modelgrid import *
#import observate

class CSP(SpecLibrary):

    def __init__(self, SSP = None, filename = None):
        if SSP is not None:
            self.SSP = SSP
        if filename is not None:
            self.flux_unit = ' ' #work this out
            wave, spec, pars = self.read_model_from_fitsbinary(filename, ['AGE','MSTAR'],wavename = 'WAVELENGTH')
            self.wavelength = wave
            self.spectra = np.squeeze(spec)
            self.pars = pars
            self.pars['AGE'][0] = 1e-9 #set 0 age to 1yr
            
    def properties_at_age(self, target_age):
        """interpolate preexisting spectra to the spectrum at a given age"""
        inds, weights = self.weights_1DLinear(np.log10(self.pars['AGE']),np.log10(target_age))
        return self.combine_weighted_spectra(inds, weights), (weights * self.pars['MSTAR'][inds]).sum()

    ###
    #stuff here for generating your own CSP from an SSP and an SFH
    ###
    
    def attenuate_SSP(self, Dust):
        pass

    def convolve_for_spectra(self, SFH, ages = 0, Dust = None):
        """convolve the SFH with the SSP to obtain the spectrum of this
        component at the specified lookback time(s).  Adds dense extra time points
        (above what is in the SSPs) near break points or high derivatives in SFR. """
        
        ageprime = SFH.age_array(ages) #set up the lookback time array including special points
        sfr = SFH.SFR(ageprime) #get SFRs for these lookback times
        mass = SFH.integrated_SFR(ageprime)
        pars = np.vstack([ageprime,sfr,mass])
        self.pars = self.array_to_struct(pars,['AGE','SFR','MFORMED'])
        
        #interpolate the SSPs to the right lookback times if necessary
