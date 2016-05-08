import numpy as np
from scipy.interpolate import interp1d
from .parameters import ProspectorParams
try:
    from astropy.cosmology import WMAP9 as cosmo
except(ImportError):
    pass

__all__ = ["SedModel", "CSPModel"]

lsun = 3.846e33  # ergs/s
pc = 3.085677581467192e18  # cm
jansky_mks = 1e-26
#value to go from L_sun/Hz to erg/s/cm^2/Hz at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

class SedModel(ProspectorParams):
    """
    For models composed of SSPs and sums of SSPs which use the
    sps_basis.StellarPopBasis as the sps object.
    """

    def mean_model(self, theta, obs, sps=None, **extras):
        """
        Given a theta vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        :param theta:
            ndarray of parameter values.

        :param obs:
            An observation dictionary, containing the output
            wavelength array, the photometric filter lists, and the
            key 'logify_spectrum' which is True if the comparison to
            the model is to be made in the log.
            
        :param sps:
            A StellarPopBasis object to be used
            in the model generation.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by obs['wavelength'], and optionally in the log.
            
        :returns phot:
            The model photometry for these parameters, for the filters
            specified in obs['filters'].
            
        :returns extras:
            Any extra aspects of the model that are returned.
        """
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        if obs.get('logify_spectrum', True):
            s = np.log(s) + np.log(self.spec_calibration(obs=obs, **extras))
        else:
            s *= self.spec_calibration(obs=obs, **extras)
        return s, p, x
    
    def sed(self, theta, obs, sps=None, peraa=False, **kwargs):
        """
        Given a theta vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), ***not** including any instrument
        calibration effects.

        :param theta:
            ndarray of parameter values.
            
        :param sps:
            A StellarPopBasis object to be used
            in the model generation.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by obs['wavelength'], in linear units.
            
        :returns phot:
            The model photometry for these parameters, for the filters
            specified in obs['filters'].
            
        :returns extras:
            Any extra aspects of the model that are returned.
        """
        
        self.set_parameters(theta)        
        spec, phot, extras = sps.get_spectrum(outwave=obs['wavelength'],
                                              filters=obs['filters'],
                                              peraa=peraa,
                                              **self.params)
        
        spec *= obs.get('normalization_guess', 1.0)
        #remove negative fluxes
        try:
            tiny = 1.0/len(spec) * spec[spec > 0].min()
            spec[ spec < tiny ] = tiny
        except:
            pass
        spec = (spec + self.sky())
        return spec, phot, extras

    def sky(self):
        """Model for the *additive* sky emission/absorption"""
        return 0.
        
    def spec_calibration(self, theta=None, obs=None, **kwargs):
        """
        Implements a polynomial calibration model.  This only happens
        if `pivot_wave` is a key of the obs dictionary, since the
        polynomial is defined in terms of r'$x \equiv
        \lambda/\lambda_{{pivot}} - 1$'.

        :returns cal:
           a polynomial given by 'spec_norm' * (1 + \Sum_{m=1}^M
           'poly_coeffs'[m-1] x**m)
        """
        if theta is not None:
            self.set_parameters(theta)
        
        # Should find a way to make this more generic,
        # using chebyshev polynomials
        if 'pivot_wave' in obs:
            x = obs['wavelength']/obs['pivot_wave'] - 1.0
            poly = np.zeros_like(x)
            powers = np.arange( len(self.params['poly_coeffs']) ) + 1
            poly = (x[None,:] ** powers[:,None] *
                    self.params['poly_coeffs'][:,None]).sum(axis = 0)
            #switch to have spec_norm be multiplicative or additive
            #depending on whether the calibration model is
            #multiplicative in exp^poly or just poly, Should move this
            #to mean_model()?
            if self.params.get('cal_type', 'exp_poly') is 'poly':
                return (1.0 + poly) * self.params['spec_norm']
            else:
                return np.exp(self.params['spec_norm'] + poly)
        else:
            return 1.0

    def spec_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        pars = ['gp_jitter', 'gp_amplitude', 'gp_length']
        defaults = [[0.0], [0.0], [1.0]]
        vals = [self.params.get(p, d) for p, d in zip(pars, defaults)]
        return  tuple(vals)
    
    def phot_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        s = self.params.get('phot_jitter', 0.0)
        return s, [0.0], [0]

    
class CSPModel(ProspectorParams):
    """
    For parameterized SFHs where fsps.StellarPopulation is used as the
    sps object.
    """

    def mean_model(self, theta, obs, sps=None, **kwargs):
        """Rename of CSPModel.sed() for compatibility.  If any
        parameteric calibration model is applied, it should go here.
        """
        return self.sed(theta, obs, sps=sps, **kwargs)
    
    def sed(self, theta, obs, sps=None, **kwargs):
        """
        Given a theta vector, generate spectroscopy, photometry and any
        extras (e.g. stellar mass).

        :param theta:
            ndarray of parameter values.

        :param sps:
            A python-fsps StellarPopulation object to be used for
            generating the SED.

        :returns spec:
            The restframe spectrum in units of L_\odot/Hz
            
        :returns phot:
            The apparent (redshifted) maggies in each of the filters.

        :returns extras:
            A list of None type objects, only included for consistency
            with the SedModel class.
        """
        self.set_parameters(theta)
        # Pass the model parameters through to the sps object
        ncomp = len(self.params['mass'])
        for ic in range(ncomp):
            s, p, x = self.one_sed(component_index = ic, sps=sps,
                                   filterlist=obs['filters'])
            try:
                spec += s
                maggies += p
                extra += [x]
            except(NameError):
                spec, maggies, extra = s, p, [x]
                
        if obs['wavelength'] is not None:
            spec = interp1d( sps.wavelengths, spec, axis = -1,
                             bounds_error=False)(obs['wavelength'])
                
        #modern FSPS does the distance modulus for us in get_mags,
        # !but python-FSPS does not!
        if sps.params['zred'] == 0:
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('dist', 1e-5) * 1e5)**2
            # spectrum stays in L_sun/Hz
            dfactor_spec = 1.0
        else:
            dfactor = ((cosmo.luminosity_distance(sps.params['zred']).value *
                        1e5)**2 / (1+sps.params['zred']))
            # convert to maggies
            dfactor_spec = to_cgs / 1e3 / dfactor / (3631*jansky_mks)
        return (spec * dfactor_spec + self.sky(),
                maggies / dfactor,
                extra)

    def one_sed(self, component_index=0, sps=None, filterlist=[]):
        """Get the SED of one component for a multicomponent composite
        SFH.  Should set this up to work as an iterator.
        
        :param component_index:
            Integer index of the component to calculate the SED for.
            
        :params sps:
            A python-fsps StellarPopulation object to be used for
            generating the SED.

        :param filterlist:
            A list of strings giving the (FSPS) names of the filters
            onto which the spectrum will be projected.
            
        :returns spec:
            The restframe spectrum in units of L_\odot/Hz.
            
        :returns maggies:
            Broadband fluxes through the filters named in
            ``filterlist``, ndarray.  Units are observed frame
            absolute maggies: M = -2.5 * log_{10}(maggies).
            
        :returns extra:
            The extra information corresponding to this component.
        """
        # Pass the model parameters through to the sps object,
        # and keep track of the mass of this component
        mass = 1.0
        for k, vs in list(self.params.items()):
            try:
                v = vs[component_index]
                #n_param_is_vec += 1
            except(IndexError, TypeError):
                v = vs
            if k in sps.params.all_params:
                if k == 'zmet':
                    vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
                else:
                    vv = v.copy()
                sps.params[k] = vv
            if k == 'mass':
                mass = v
        #now get the magnitudes and spectrum
        w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
        mags = sps.get_mags(tage=sps.params['tage'],
                            bands=filterlist)
        # normalize by (current) stellar mass and get correct units (distance_modulus)
        mass_norm = mass/sps.stellar_mass
        return (mass_norm * spec,
                mass_norm * 10**(-0.4*(mags)),
                sps.stellar_mass)

    def phot_calibration(self, **extras):
        return 1.0

    def phot_gp_params(self, obs=None, outlier=False, theta=None,
                       **extras):
        """Return the parameters for generating the covariance matrix
        used by the photometric gaussian process in a way
        understandable for the GP objects.  This method looks for the
        ``phot_jitter``, ``gp_phot_amps`` and ``gp_phot_locs`` keys in
        the parameter dictionary

        :param obs:
            obs data dictionary.  Must have a ``filters`` key.
            
        :param outlier:
            Switch to interpret the gp parameters in terms of outlier
            modeling.  In this case ``gp_phot_locs`` is an array of
            indices into the (masked) maggies array, and
            ``gp_phot_amps`` is an array of GP amplitudes
            corresponding to these indices.

            Otherwise, ``gp_phot_locs`` is a fixed array of filter
            name lists, and ``gp_phot_amps`` is an array of the same
            length giving the GP amplitudes associated with each list

        :returns s:
            The jitter (scalar)

        :returns amps:
            The GP amplitudes, ndarray.

        :returns locs:
            Indices into the obs['maggies'][obs['phot_mask']] array
            corresponding elementwise to the returned ``amps`` array
        """

        if theta is not None:
            self.set_parameters(theta)
            
        mask = obs.get('phot_mask', np.ones( len(obs['filters']), dtype= bool))
        pars = ['phot_jitter', 'gp_phot_amps', 'gp_phot_locs']
        defaults = [0.0, [0.0], [0]]
        s, amps, locs = [self.params.get(p, d) for p, d in zip(pars, defaults)]
        no_locs = 'gp_phot_locs' not in self.params
        #outlier modeling allows the locations to float. locs here are indices
        if outlier or no_locs:
            locs = np.clip(locs, 0, mask.sum() - 1)
            return s, np.atleast_1d(amps), np.atleast_1d(locs)
        
        # Here we add linked amplitudes based on filter names.  locs
        # here is an array of lists
        ll, aa = [], []
        if type(obs['filters'][0]) is str:
            fnames = [f for i,f in enumerate(obs['filters']) if mask[i]]
        else:
            fnames = [f.name for i,f in enumerate(obs['filters']) if mask[i]]
        for i, a in enumerate(amps):
            filts = locs[i]
            ll += [fnames.index(f) for f in filts if f in fnames]
            aa += len(filts) * [a] 
        return  s, np.atleast_1d(aa), np.atleast_1d(ll)

    def sky(self):
        return 0.


def gauss(x, mu, A, sigma):
    """
    Sample multiple gaussians at positions x.

    :param x:
        locations where samples are desired.

    :param mu:
        Center(s) of the gaussians.
        
    :param A:
        Amplitude(s) of the gaussians, defined in terms of total area.

    :param sigma:
        Dispersion(s) of the gaussians, un units of x.

    :returns val:
        The values of the sum of gaussians at x 
    """ 
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)
