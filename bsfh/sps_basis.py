import numpy as np
from scipy.interpolate import interp1d
import fsps
from sedpy.observate import getSED, vac2air, air2vac

lsun = 3.846e33
pc = 3.085677581467192e18
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

class StellarPopBasis(object):
    """
    A class that wraps the python-fsps StellarPopulation object in
    order to include more functionality and to allow 'fast' model
    generation in some situations by storing an easily accessible
    spectral grid.
    """
    
    def __init__(self, smooth_velocity=True, debug=False):

        self.debug = debug
        
        #this is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(smooth_velocity = smooth_velocity)
        
        #This is the main state vector for the model
        self.params = {'dust_tesc':0.00, 'dust1':0., 'dust2':0.}
        
        #These are the parameters whose change will force a regeneration of the SSPs (and basis) using fsps
        self.ssp_params = ['imf_type','imf3','agb_dust']
        self.ssp_dirty = True
        
        #These are the parameters whose change will force a regeneration of the basis from the SSPs
        self.basis_params = ['sigma_smooth','tage','zmet','dust1','dust2','dust_tesc','zred','outwave','dust_curve']
        self.basis_dirty =True
        
    def get_spectrum(self, params, outwave, filters, nebular=True):
        """
        Return a spectrum for the given parameters.  If necessary the
        SSPs are updated, and if necessary the component spectra are
        updated, before being combined here.

        :param params:
            A dictionary-like of the model parameters.
            
        :param outwave: 
            The output wavelength points at which model estimates are
            desired, ndarray of shape (nwave,)
            
        :param filters:
             A list of filters in which synthetic photometry is
             desired.  List of length (nfilt,)

        :param nebular: (Default: True)
            If True, add a nebular spectrum to the total spectrum.
            Note that this is currently not added to the photometry as
            well

        :returns spec:
            The spectrum at the wavelength points given by outwave,
            ndarray of shape (nwave,).  Units are erg/s/cm^2/AA
            
        :returns phot:
            The synthetc photometry through the provided filters,
            ndarray of shape (nfilt,).  Note, the units are *apparent
            maggies*.

        :returns extra:
            Any extra parameters (like stellar mass) that you want to
            return.
        """
        cspec, neb, cphot, cextra = self.get_components(params, outwave, filters)

        spec = (cspec * params['mass'][:,None]).sum(axis = 0)
        if nebular:
            spec += neb
            
        #phot  = 10**(-0.4 *getSED( outwave, spec, filters)) #can't do because wavelngth grid is truncated
        phot = (cphot * params['mass'][:,None]).sum(axis = 0)
        extra = (cextra * params['mass']).sum()
        
        return spec, phot, extra
    
    def get_components(self, params, outwave, filters):
        """
        Return the component spectra for the given parameters, making
        sure to update the components if necessary.

        :param params:
            A dictionary-like of the model parameters.
            
        :param outwave: 
            The output wavelength points at which model estimates are
            desired, ndarray of shape (nwave,)
            
        :param filters:
             A list of filters in which synthetic photometry is
             desired.  List of length (nfilt,)

        :returns cspec:
            The spectrum at the wavelength points given by outwave,
            ndarray of shape (ncomp,nwave).  Units are
            erg/s/cm^2/AA/M_sun
            
        :returns phot:
            The synthetc photometry through the provided filters,
            ndarray of shape (ncomp,nfilt).  Units are
            *apparent maggies*.

        :returns extra:
            Any extra parameters (like stellar mass) that you want to
            return.
        """
        
        params['outwave'] = outwave
        self.update(params)

        #distance dimming and conversion from Lsun/AA to cgs
        dist10 = self.params.get('lumdist', 1e-5)/1e-5 #distance in units of 10s of pcs
        dfactor = to_cgs / dist10**2
        
        # Stellar component. Redshift and put on the proper wavelength
        # grid. Eventually this should probably do proper integration
        # within the output wavelength bins, and deal with non-uniform
        # line-spread functions
        a1 = (1 + self.params.get('zred', 0.0))
        cspec = interp1d( vac2air(self.ssp.wavelengths * a1),
                          self.basis_spec / a1 * dfactor, axis = -1)(outwave)

        # Nebular component.  Should add this to spectra somehow
        # before generating photometry.
        neb = self.nebular(params, outwave) * dfactor
        
        #get the photometry
        cphot = 10**(-0.4 *getSED( self.ssp.wavelengths * a1,
                                   self.basis_spec / a1 * dfactor, filters))

        
        return cspec, neb, cphot, self.basis_mass
    
    def nebular(self, params, outwave):
        """
        If the emission_rest_wavelengths parameter is present, return
        a nebular emission line spectrum.  Currently uses several
        approximations for the velocity broadening.  Currently does
        *not* affect photometry.  Only provides samples of the nebular
        spectrum at outwave, so will not be correct for total power
        unless outwave densley samples the emission dispersion.

        :returns nebspec:
            The nebular emission in the observed frame, at the wavelengths
            specified by the obs['wavelength'].
        """

        if 'emission_rest_wavelengths' in params:
            mu = vac2air(params['emission_rest_wavelengths'])
            # try to get a nebular redshift, otherwise use stellar
            # redshift, otherwise use no redshift
            a1 = params.get('zred_emission', self.params.get('zred', 0.0)) + 1.0
            A =  params.get('emission_luminosity',0.)
            sigma = params.get('emission_disp',10.)
            if params.get('smooth_velocity', False):
                #This is an approximation to get the dispersion in
                # terms of wavelength at the central line wavelength,
                # but should work much of the time
                sigma = mu * sigma / 2.998e5
            return gauss(outwave, mu * a1, A, sigma * a1)
        
        else:
            return 0.


    def build_basis(self, outwave):
        """
        Rebuild the component spectra from the SSPs.  The component
        spectra include dust attenuation, redshifting, and spectral
        regridding.  This is basically a proxy for COMPSP from FSPS,
        with a few small differences.  In particular, there is
        interpolation in metallicity and the redshift and the output
        wavelength grid are taken into account.  The dust treatment is
        less sophisticated.

        This method is only called by self.update if necessary.

        :param outwave: 
            The output wavelength points at which model estimates are
            desired, ndarray of shape (nwave,)

        """
        #setup the internal component basis arrays
        inwave = self.ssp.wavelengths
        nbasis = len(np.atleast_1d(self.params['zmet'])) * len(np.atleast_1d(self.params['tage']))
        self.basis_spec = np.zeros([nbasis, len(inwave)])
        self.basis_mass = np.zeros(nbasis)

        i = 0
        #should vectorize this set of loops
        for j,zmet in enumerate(self.params['zmet']):
            for k,tage in enumerate(self.params['tage']):
                # get the intrinsic spectrum at this metallicity and age
                spec, mass, lbol = self.ssp.ztinterp(zmet, tage, peraa = True)
                
                # and attenuate by dust unless missing any dust parameters
                # This is ugly - should use a hook into ADD_DUST
                dust = ((tage < self.params['dust_tesc']) * self.params['dust1']  +
                        (tage >= self.params['dust_tesc']) * self.params['dust2'])
                spec *= np.exp(-self.params['dust_curve'][0](inwave) * dust)
                # broaden and store
                self.basis_spec[i,:] = self.ssp.smoothspec(inwave, spec, self.params.get('sigma_smooth',0.0))
                # = griddata(inwave * a1, spec / a1, self.params['outwave'])
                self.basis_mass[i] = mass
                i += 1
                
        self.basis_dirty = False

    def update(self, inparams):
        """
        Update the parameters, recording whether it was new for the
        ssp or basis parameters.  If either of those changed,
        regenerate the relevant spectral grid(s).
        """
        
        for k,v in inparams.iteritems():
            if k in self.ssp_params:
                try:
                    #here the sps.params.dirtiness should increase to 2 if there was a change
                    self.ssp.params[k] = v
                except KeyError:
                    pass
            elif k in self.basis_params:
                if self.debug:
                    print(k, np.any(v != self.params.get(k,None)))
                if np.any(v != self.params.get(k,None)):
                    self.basis_dirty = True
            #now update params
            self.params[k] = np.copy(np.atleast_1d(v))

        if self.basis_dirty | (self.ssp.params.dirtiness == 2):
            self.build_basis(inparams['outwave'])


def gauss(x, mu, A, sigma):
    """
    Lay down mutiple gaussians on the x-axis.
    """ 
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)

            

def selftest():
    from sedpy.observate import load_filters
    sps = sps_basis.StellarPopBasis(debug =True)
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
