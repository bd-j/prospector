import numpy as np

from observate import getSED
from scipy.interpolate import griddata
from simplemodel import Model

lsun, pc = 3.846e33, 3.085677581467192e18 #in cgs
to_cgs = lsun/10**( np.log10(4.0*np.pi)+2*np.log10(pc*10) )

class ClusterModel(Model):

    def __call__(self, theta):
        return self.lnprob(theta, sps = sps)
          
    def __init__(self, theta_desc, filters = None):
        self.verbose = False
        self.filters = filters
        self.theta_desc = theta_desc
        #self.sps = sps
        #self.sps_params = {}
        #self.sps_params['sfh'] = 0
        #self.sps_params['imf_type'] = 2
        #self.sps_params['dust_type'] = 1
        self.jitter = 0
        self.ssp = {'mass':0., 'zred':0.0}
        self.cal_pars = {'pivot_wave':5500., 'spec_norm':1.0}
        self.compsp_pars_from_theta = ['mass']
        self.ssp_pars_from_theta = ['tage', 'zmet', 'imf3', 'dust2', 'vel_broad', 'zred'] #['imf1','f_bhb'] #etc...
        self.cal_pars_from_theta = ['poly_coeffs', 'spec_norm']#['poly_coeffs', 'jitter'] #etc....

    def set_parameters(self, theta, sps = None):
        """Propogate theta into the model parameters"""

        # Parameters for compsp - these are propogated into the parameters for each component
        for p in self.compsp_pars_from_theta:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            self.ssp[p] = np.array(theta[start:end])
        # Parameters for the calibration model.  these are stored in cal_pars       
        for p in self.cal_pars_from_theta:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            self.cal_pars[p] = np.array(theta[start:end])
        # Parameters for the SSP generation.  These are propogated through
        #  to the sps object and should be FSPS params
        for p in self.ssp_pars_from_theta:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            if p == 'zmet':
                #print(theta[start:end], np.clip(np.round(theta[start:end]),1,5))
                tmp = np.clip(np.round(theta[start:end]),1,5)
            else:
                tmp = theta[start:end]
            sps.params[p] = tmp[0] #this should increase dirtiness and force a regeneration of the SSPs
            #print(self.sps.params.dirtiness)
            
    def model(self, theta, sps = None, filters = None):
        """Given a theta vector, generate a spectrum, photometry, and any extras."""

        self.set_parameters(theta, sps = sps) # Propogate theta into the relevant parameter arrays
        wave, spec =  sps.get_spectrum(peraa = True, tage = sps.params['tage'])
        spec *= self.ssp['mass']/sps.stellar_mass
        try:
            z1 = (1 + self.ssp['zred'])
        except (KeyError):
            z1 = 1.0
        try:
            outspec = griddata(wave * z1, spec, self.obs['wavelength']) * self.calibration() 
        except (KeyError):
            outspec = spec
        
        if self.filters is not None:
            filters = self.filters #Bad pattern
        if filters is not None:
            #SLOOOOWWW.  also, this affects the spectrum for subsequent calls to get_spectrum()
            #phot = self.ssp['mass'] * 10**(-0.4 * sps.get_mags(tage = sps.params['tage'],
            #                                bands = filters, redshift = sps.params['zred'])) 
            phot = 10.**(-0.4 * getSED(wave * z1, spec / z1 * to_cgs, filters))
        else:
            phot = None
        others = None
        return outspec, phot, others
        
    def calibration(self):
        x = self.obs['wavelength']/self.cal_pars['pivot_wave'] - 1.0
        poly = np.zeros_like(x)
        powers = np.arange( len(self.cal_pars['poly_coeffs']) ) + 1
        poly = (x[None,:] ** powers[:,None] * self.cal_pars['poly_coeffs'][:,None]).sum(axis = 0)
        
        return (1.0 + poly) * self.cal_pars['spec_norm']
