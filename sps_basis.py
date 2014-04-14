import numpy as np
from scipy.interpolate import griddata
import astropy.constants as constants
from observate import getSED
from sfhutils import weights_1DLinear

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

class StellarPopBasis(object):

    def __init__(self, sps):
        #this is a StellarPopulation object from fsps
        self.sps = sps
        self.ssp_zlegend = np.array([0.008, 0.0031, 0.0096, 0.0190, 0.0300])
        #This is the main state vector for the model
        self.params = {'dust_tesc':0.02, 'dust1':0., 'dust2':0.}
        #These are the parameters whose change will force a regeneration of the SSPs (and basis) using fsps
        self.ssp_params = ['vel_broad','sigma_smooth','imf_type','imf3','agb_dust']
        #These are the parameters whose change will force a regeneration of the basis from the SSPs
        self.basis_params = ['tage','zmet','dust1','dust2','dust_tesc','zred','outwave','dust_curve']
        self.ssp_dirty = True
        self.basis_dirty =True
        
    def get_spectrum(self, inparams, outwave, filters):
        #return a spectrum for the given parameters
        cspec, cphot, cextra = self.get_components(inparams, outwave, filters)

        spec = (cspec * inparams['mass'][:,None]).sum(axis = 0)    
        phot = (cphot * inparams['mass'][:,None]).sum(axis = 0)
        extra = (cextra * inparams['mass']).sum()
        return spec, phot, extra
    
    def get_components(self, inparams, outwave, filters):
        inparams['outwave'] = outwave
        self.update(inparams)            
        return self.basis_spec, getSED(self.basis_wave, self.basis_spec * to_cgs, filters), self.basis_mass

    def build_ssp(self):
        """Rebuild the SSPs.  This is basically a proxy/wrapper for
        SSP_GEN from FSPS.
        """
        for i, zmet in enumerate(self.ssp_zlegend):
            wave, spec = self.sps.get_spectrum(peraa =True, tage = 0., zmet = i+1)
            if i == 0:
                self.ssp_spec = np.zeros([ len(self.ssp_zlegend), spec.shape[0], spec.shape[1] ])
                self.ssp_mass = np.zeros([ len(self.ssp_zlegend), spec.shape[0] ])
            self.ssp_spec[i,:,:] = spec
            self.ssp_mass[i,:] = self.sps.stellar_mass
        self.ssp_logage = self.sps.log_age - 9. #in log Gyr
        self.ssp_dirty = False
        
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
        inwave = self.sps.wavelengths
        #redshift
        z1 = (1 + self.params.get('zred', 0.0))
        self.basis_wave = self.params['outwave']
        #should vectorize this
        for j,zmet in enumerate(self.params['zmet']):
            for k,tage in enumerate(self.params['tage']):
                #get the intrinsic spectrum at this metallicity and age
                spec, mass = self.ztinterp(zmet, tage)
                #print(spec.shape, mass.shape)
                #and attenuate by dust unless missing any dust parameters
                try:
                    dust = ((tage < self.params['dust_tesc']) * self.params['dust1']  +
                            (tage >= self.params['dust_tesc']) * self.params['dust2'])
                    spec *= np.exp(-self.params['dust_curve'](inwave))
                except KeyError:
                    pass
                #redshift and put on the proper wavelength grid
                #eventually this should probably do proper integration within
                # the output wavelength bins.  It should also allow for 
                self.basis_spec[i,:] = griddata(inwave * z1, spec/z1, self.params['outwave'] )
                self.basis_mass[i] = mass
                i += 1
        self.basis_dirty = False

    def ztinterp(self, zmet, tage):
        """Bilinear interpolation in log age and log Z/Zsun"""
        ainds, aweights = weights_1DLinear(self.ssp_logage, [np.log10(tage)])
        zinds, zweights = weights_1DLinear(np.log10(self.ssp_zlegend/0.0190), [zmet])
        spec = (self.ssp_spec[:,ainds[0,:],:] * aweights[:,:,None]).sum(axis = 1)
        mass = (self.ssp_mass[:,ainds[0,:]] * aweights).sum(axis = 1)
        spec = (spec[zinds[0,:],:] * zweights.T).sum(axis = 0)
        mass = (mass[zinds[0,:]] * zweights[0,:]).sum()
        return spec, mass
    
    def update(self, inparams):
        """Update the parameters, recording whether it was new
        for the ssp or basis parameters.  If it was, regenerate the
        relevant spectral grid
        """
        ssp_dirty, basis_dirty = False, False
        for k,v in inparams.iteritems():
            if k in self.ssp_params:
                if np.any(v != self.params.get(k,None)):
                    print(k)
                    ssp_dirty = True
                    try:
                        self.sps.params[k] = v
                    except KeyError:
                        pass
            elif k in self.basis_params:
                print(k, np.any(v != self.params.get(k,None)))
                if np.any(v != self.params.get(k,None)):
                    basis_dirty = True
        
            self.params[k] = np.copy(np.atleast_1d(v))
        self.ssp_dirty = self.ssp_dirty | ssp_dirty
        self.basis_dirty = self.basis_dirty | basis_dirty
        print(self.ssp_dirty, self.basis_dirty)
        if self.ssp_dirty:
            self.build_ssp()
        if self.ssp_dirty or self.basis_dirty:
            self.build_basis(inparams['outwave'])



def selftest():
    import fsps
    from observate import load_filters
    sp = fsps.StellarPopulation()
    sps = StellarPopBasis(sps)
    params = {}
    params['tage'] = np.array([1,2,3,4.])
    params['zmet'] = np.array([-0.5,0.0])
    params['mass'] = np.random.uniform(0,1,len(params['tage']) * len(params['zmet']))

    outwave = sp.wavelengths
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
