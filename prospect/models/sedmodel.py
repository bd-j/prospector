import numpy as np
from scipy.interpolate import interp1d
from .parameters import ProspectorParams
try:
    from astropy.cosmology import WMAP9 as cosmo
except(ImportError):
    pass

__all__ = ["SedModel"]

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
    
    def sed(self, theta, obs, sps=None, **kwargs):
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
