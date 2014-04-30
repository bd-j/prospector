import numpy as np
from scipy.interpolate import griddata
import astropy.constants as constants
import fsps
from observate import getSED, broaden
from sfhutils import weights_1DLinear

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

class StellarPopBasis(object):

    def __init__(self, smooth_velocity = True):
        #this is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(smooth_velocity = smooth_velocity)
        
        #This is the main state vector for the model
        self.params = {'dust_tesc':0.02, 'dust1':0., 'dust2':0.}
        
        #These are the parameters whose change will force a regeneration of the SSPs (and basis) using fsps
        self.ssp_params = ['imf_type','imf3','agb_dust']
        self.ssp_dirty = True
        
        #These are the parameters whose change will force a regeneration of the basis from the SSPs
        self.basis_params = ['sigma_smooth','tage','zmet','dust1','dust2','dust_tesc','zred','outwave','dust_curve']
        self.basis_dirty =True
        
    def get_spectrum(self, inparams, outwave, filters):
        """
        Return a spectrum for the given parameters.
        """
        cspec, cphot, cextra = self.get_components(inparams, outwave, filters)

        spec = (cspec * inparams['mass'][:,None]).sum(axis = 0)    
        phot = (cphot * inparams['mass'][:,None]).sum(axis = 0)
        extra = (cextra * inparams['mass']).sum()
        return spec, phot, extra
    
    
    def get_components(self, inparams, outwave, filters):
        """
        Return the component spectra for the given parameters,
        making sure to update the components if necessary
        """

        inparams['outwave'] = outwave
        self.update(inparams)            
        return self.basis_spec, getSED(self.basis_wave, self.basis_spec * to_cgs, filters), self.basis_mass
    

    def build_basis(self, outwave):
        """ Rebuild the component spectra from the SSPs.  The component
        spectra include dust attenuation, redshifting, and spectral regridding.
        This is basically a proxy for COMPSP from FSPS, with a few small differences.
        In particular, there is interpolation in metallicity and the redshift and
        the output wavelength grid are taken into account.  The dust treatment is
        less sophiticated.
        """
        #setup the internal component basis arrays
        nbasis = len(np.atleast_1d(self.params['zmet'])) * len(np.atleast_1d(self.params['tage']))
        self.basis_spec = np.zeros([nbasis, len(self.params['outwave'])])
        self.basis_mass = np.zeros(nbasis)
        i = 0
        inwave = self.ssp.wavelengths
        
        #scale factor from redshift
        a1 = (1 + self.params.get('zred', 0.0))
        self.basis_wave = self.params['outwave']
        
        #should vectorize this set of loops
        for j,zmet in enumerate(self.params['zmet']):
            for k,tage in enumerate(self.params['tage']):
                #get the intrinsic spectrum at this metallicity and age
                spec, mass, lbol = self.ssp.ztinterp(zmet, tage, peraa = True)
                
                #and attenuate by dust unless missing any dust parameters
                #This is ugly - should use a hook into ADD_DUST
                try:
                    dust = ((tage < self.params['dust_tesc']) * self.params['dust1']  +
                            (tage >= self.params['dust_tesc']) * self.params['dust2'])
                    spec *= np.exp(-self.params['dust_curve'][0](inwave))
                except KeyError:
                    pass
                # Redshift and put on the proper wavelength grid
                # Eventually this should probably do proper integration within
                # the output wavelength bins.
                spec = self.ssp.smoothspec(inwave, spec, self.params.get('sigma_smooth',0.0))
                self.basis_spec[i,:] = griddata(inwave * a1, spec / a1, self.params['outwave'])
                self.basis_mass[i] = mass
                i += 1
                
        self.basis_dirty = False

        
    def update(self, inparams):
        """Update the parameters, recording whether it was new
        for the ssp or basis parameters.  If those changed,
        regenerate the relevant spectral grid(s).
        """
        ssp_dirty, basis_dirty = False, False
        for k,v in inparams.iteritems():
            if k in self.ssp_params:
                try:
                    #here the sps.params.dirtiness should increase to 2 if there was a change
                    self.ssp.params[k] = v
                except KeyError:
                    pass
            elif k in self.basis_params:
                #print(k, np.any(v != self.params.get(k,None)))
                if np.any(v != self.params.get(k,None)):
                    self.basis_dirty = True
            #now update params
            self.params[k] = np.copy(np.atleast_1d(v))
            
        if self.basis_dirty | (self.ssp.params.dirtiness == 2):
            self.build_basis(inparams['outwave'])


def selftest():
    from observate import load_filters
    sps = sps_basis.StellarPopBasis()
    params = {}
    params['tage'] = np.array([1,2,3,4.])
    params['zmet'] = np.array([-0.5,0.0])
    params['mass'] = np.random.uniform(0,1,len(params['tage']) * len(params['zmet']))
    params['sigma_smooth'] = 100.
    outwave = sps.ssp.wavelengths
    flist = ['sdss_u0', 'sdss_r0']
    filters = load_filters(flist)

    #get a spectrum
    s, p, e = sps.get_spectrum(params, outwave, filters)
    #change parameters that affect neither the basis nor the ssp, and get spectrum again
    params['mass'] = np.random.uniform(0,1,len(params['tage']) * len(params['zmet']))
    s, p, e = sps.get_spectrum(params, outwave, filters)
    #lets get the basis components while we're at it
    bs, bp, be = sps.get_components(params, outwave, filters)
    #change something that affects the basis
    params['tage'] += 1.0
    bs, bp, be = sps.get_components(params, outwave, filters)
    #try a single age pop at arbitrary metallicity
    params['tage'] = 1.0
    params['zmet'] = -0.2
    bs, bp, be = sps.get_components(params, outwave, filters)
