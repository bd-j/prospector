import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander
from .parameters import ProspectorParams

__all__ = ["SedModel", "SpecModel", "PolySedModel"]

class SedModel(ProspectorParams):
    """A subclass of :py:class:`ProspectorParams` that passes the models
    through to an ``sps`` object and returns spectra and photometry, including
    optional spectroscopic calibration and sky emission.
    """

    def mean_model(self, theta, obs, sps=None, **extras):
        """Given a ``theta`` vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        :param theta:
            ndarray of parameter values, of shape ``(ndim,)``

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the key ``"logify_spectrum"`` which
            is ``True`` if the comparison to the model is to be made in the log.

        :param sps:
            An `sps` object to be used in the model generation.  It must have
            the :py:method:`get_spectrum` method defined.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by ``obs['wavelength']``, including multiplication by the
            calibration vector.

        :returns phot:
            The model photometry for these parameters, for the filters
            specified in ``obs['filters']``.  Units of maggies.

        :returns extras:
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            stellar mass formed.
        """
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        self._speccal = self.spec_calibration(obs=obs, **extras)
        if obs.get('logify_spectrum', False):
            s = np.log(s) + np.log(self._speccal)
        else:
            s *= self._speccal
        return s, p, x

    def sed(self, theta, obs, sps=None, **kwargs):
        """Given a ``theta vector``, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), ***not** including any instrument
        calibration effects.

        :param theta:
            ndarray of parameter values.

        :param sps:
            A StellarPopBasis object to be used
            in the model generation.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by ``obs['wavelength']``.  Default units are maggies, and
            the calibration vector is **not** applied.

        :returns phot:
            The model photometry for these parameters, for the filters
            specified in ``obs['filters']``. Units are maggies.

        :returns extras:
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            steallr mass formed.
        """
        self.set_parameters(theta)
        spec, phot, extras = sps.get_spectrum(outwave=obs['wavelength'],
                                              filters=obs['filters'],
                                              component=obs.get('component', -1),
                                              lnwavegrid=obs.get('lnwavegrid', None),
                                              **self.params)

        spec *= obs.get('normalization_guess', 1.0)
        # Remove negative fluxes.
        try:
            tiny = 1.0/len(spec) * spec[spec > 0].min()
            spec[spec < tiny] = tiny
        except:
            pass
        spec = (spec + self.sky(obs))
        self._spec = spec.copy()
        return spec, phot, extras

    def sky(self, obs):
        """Model for the *additive* sky emission/absorption"""
        return 0.

    def spec_calibration(self, theta=None, obs=None, **kwargs):
        """Implements a Chebyshev polynomial calibration model.  This only
        occurs if ``"poly_coeffs"`` is present in the :py:attr:`params`
        dictionary, otherwise the value of ``params["spec_norm"]`` is returned.

        :param theta: (optional)
            If given, set :py:attr:`params` using this vector before
            calculating the calibration polynomial. ndarray of shape
            ``(ndim,)``

        :param obs:
            A dictionary of observational data, must contain the key
            ``"wavelength"``

        :returns cal:
           If ``params["cal_type"]`` is ``"poly"``, a polynomial given by
           ``'spec_norm'`` :math:`\times (1 + \Sum_{m=1}^M```'poly_coeffs'[m-1]``:math:` \times T_n(x))`.
           Otherwise, the exponential of a Chebyshev polynomial.
        """
        if theta is not None:
            self.set_parameters(theta)

        if ('poly_coeffs' in self.params):
            mask = obs.get('mask', slice(None))
            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = self.wave_to_x(obs["wavelength"], mask)
            # get coefficients.  Here we are setting the first term to 0 so we
            # can deal with it separately for the exponential and regular
            # multiplicative cases
            c = np.insert(self.params['poly_coeffs'], 0, 0)
            poly = chebval(x, c)
            # switch to have spec_norm be multiplicative or additive depending
            # on whether the calibration model is multiplicative in exp^poly or
            # just poly
            if self.params.get('cal_type', 'exp_poly') == 'poly':
                return (1.0 + poly) * self.params.get('spec_norm', 1.0)
            else:
                return np.exp(self.params.get('spec_norm', 0) + poly)
        else:
            return 1.0 * self.params.get('spec_norm', 1.0)


    def wave_to_x(self, wavelength=None, mask=slice(None), **extras):
        """Map unmasked wavelengths to the interval -1, 1
              masked wavelengths may have x>1, x<-1
        """
        x = wavelength - (wavelength[mask]).min()
        x = 2.0 * (x / (x[mask]).max()) - 1.0
        return x

    def spec_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        pars = ['gp_jitter', 'gp_amplitude', 'gp_length']
        defaults = [[0.0], [0.0], [1.0]]
        vals = [self.params.get(p, d) for p, d in zip(pars, defaults)]
        return tuple(vals)

    def phot_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        s = self.params.get('phot_jitter', 0.0)
        return s, [0.0], [0]

class SpecModel(ProspectorParams):
    """A subclass of :py:class:`ProspectorParams` that passes the models
    through to an ``sps`` object and returns spectra and photometry, including
    optional spectroscopic calibration and sky emission.
    """

    def predict(self,theta,obs,sps=None,**extras):

        # generate and cache model spectrum and info
        self.set_parameters(theta)
        self._wave, self._spec, self._mfrac = sps.get_galaxy_spectrum(**self.params)
        self._zred = self.params.get('zred',0)
        self._eline_wave, self._eline_lum = sps.get_emlines()
        self._norm_spec = self._spec * self.flux_norm()

        # generate spectrum and photometry for likelihood
        spec = self.predict_spec(obs)
        phot = self.predict_phot(obs['filters'])

    def predict_spec(self, obs):
      
        # de-redshift wavelength
        obs_wave = self.observed_wave(self._wave, do_wavecal=True)
        self._outwave = obs.get('wavelengths',obs_wave)

        # smoothing
        smooth_spec = self.smoothspec(obs_wave)
      
        # calibration
        self._speccal = self.spec_calibration(obs=obs, **extras)
        calibrated_spec = smooth_spec * self._speccal

        # analytical emission lines
        # eline luminosities should have same size as eline lum
        dat = self.get_el(obs, calibrated_spec, eline_wave = eline_wave, eline_lum=eline_lum)
        self._eline_spectrum, self._eline_penalty = dat

        return calibrated_spec

    def predict_phot(self,filters):

        if filters is None:
            return 0.0

        # generate photometry w/o emission lines
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        mags = getSED(obs_wave, self._norm_spec * lightspeed / obs_wave**2 * (3631*jansky_cgs), filters)
        phot = np.atleast_1d(10**(-0.4 * mags))

        # generate emission-line photometry
        phot += nebline_photometry(filters)

        return phot

    def nebline_photometry(self,filters):
        """analytically calculate emission line contribution to photometry
            fixme: check units of emission line luminosities
        """
        elams = self._eline_wave * (1+self._zred)
        elums = self._eline_lum * self.flux_norm() / (1+self._zred)

        flux = np.zeros(len(filters))
        for i,filt in enumerate(filters):
            # calculate transmission at nebular emission
            trans = np.interp(elams, filt.wavelength, filt.transmission, left=0., right=0.)
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*elams[idx]*elums[idx]).sum()/filt.ab_zero_counts

        return flux


    def observed_wave(self, wave, do_wavecal=False):
        # missing wavelength calibration (add later)
        a = 1 + self._zred
        return wave*a
    
    def flux_norm(self):
      
        # distance factor
        if (self._zred == 0) | ('lumdist' in self.params): 
            lumdist = self.params.get('lumdist',1e-5)
        else:
            lumdist = cosmo.luminosity_distance(zred).to('Mpc').value
        dfactor = (lumdist * 1e5)**2

        # Mass normalization
        mass = np.sum(self.params.get('mass',1.0))
      
        # units
        unit_conversion = to_cgs / (3631*jansky_cgs) * (1+self._zred)

        return mass * unit_conversion / dfactor 
    
    def smoothspec(self,wave):
        sigma = self.params.get('sigma',100)
        outspec = smoothspec(wave, self._norm_spec, sigma, outwave=self._outwave)

        return outspec
    
    def emission_lines(self):
      
      pass
    
    def distance_norm(self):
      
      pass
    
    def get_el(self):

        pass
    
    def mean_model(self, theta, obs, sps=None, **extras):
        """Given a ``theta`` vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        :param theta:
            ndarray of parameter values, of shape ``(ndim,)``

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the key ``"logify_spectrum"`` which
            is ``True`` if the comparison to the model is to be made in the log.

        :param sps:
            An `sps` object to be used in the model generation.  It must have
            the :py:method:`get_spectrum` method defined.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by ``obs['wavelength']``, including multiplication by the
            calibration vector.

        :returns phot:
            The model photometry for these parameters, for the filters
            specified in ``obs['filters']``.  Units of maggies.

        :returns extras:
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            stellar mass formed.
        """
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        self._speccal = self.spec_calibration(obs=obs, **extras)
        if obs.get('logify_spectrum', False):
            s = np.log(s) + np.log(self._speccal)
        else:
            s *= self._speccal
        return s, p, x

    def sed(self, theta, obs, sps=None, **kwargs):
        """Given a ``theta vector``, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), ***not** including any instrument
        calibration effects.

        :param theta:
            ndarray of parameter values.

        :param sps:
            A StellarPopBasis object to be used
            in the model generation.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by ``obs['wavelength']``.  Default units are maggies, and
            the calibration vector is **not** applied.

        :returns phot:
            The model photometry for these parameters, for the filters
            specified in ``obs['filters']``. Units are maggies.

        :returns extras:
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            steallr mass formed.
        """
        self.set_parameters(theta)
        spec, phot, extras = sps.get_spectrum(outwave=obs['wavelength'],
                                              filters=obs['filters'],
                                              component=obs.get('component', -1),
                                              lnwavegrid=obs.get('lnwavegrid', None),
                                              **self.params)

        spec *= obs.get('normalization_guess', 1.0)
        # Remove negative fluxes.
        try:
            tiny = 1.0/len(spec) * spec[spec > 0].min()
            spec[spec < tiny] = tiny
        except:
            pass
        spec = (spec + self.sky(obs))
        self._spec = spec.copy()
        return spec, phot, extras

    def sky(self, obs):
        """Model for the *additive* sky emission/absorption"""
        return 0.

    def spec_calibration(self, theta=None, obs=None, **kwargs):
        """Implements a Chebyshev polynomial calibration model.  This only
        occurs if ``"poly_coeffs"`` is present in the :py:attr:`params`
        dictionary, otherwise the value of ``params["spec_norm"]`` is returned.

        :param theta: (optional)
            If given, set :py:attr:`params` using this vector before
            calculating the calibration polynomial. ndarray of shape
            ``(ndim,)``

        :param obs:
            A dictionary of observational data, must contain the key
            ``"wavelength"``

        :returns cal:
           If ``params["cal_type"]`` is ``"poly"``, a polynomial given by
           ``'spec_norm'`` :math:`\times (1 + \Sum_{m=1}^M```'poly_coeffs'[m-1]``:math:` \times T_n(x))`.
           Otherwise, the exponential of a Chebyshev polynomial.
        """
        if theta is not None:
            self.set_parameters(theta)

        if ('poly_coeffs' in self.params):
            mask = obs.get('mask', slice(None))
            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = self.wave_to_x(obs["wavelength"], mask)
            # get coefficients.  Here we are setting the first term to 0 so we
            # can deal with it separately for the exponential and regular
            # multiplicative cases
            c = np.insert(self.params['poly_coeffs'], 0, 0)
            poly = chebval(x, c)
            # switch to have spec_norm be multiplicative or additive depending
            # on whether the calibration model is multiplicative in exp^poly or
            # just poly
            if self.params.get('cal_type', 'exp_poly') is 'poly':
                return (1.0 + poly) * self.params.get('spec_norm', 1.0)
            else:
                return np.exp(self.params.get('spec_norm', 0) + poly)
        else:
            return 1.0 * self.params.get('spec_norm', 1.0)


    def wave_to_x(self, wavelength=None, mask=slice(None), **extras):
        """Map unmasked wavelengths to the interval -1, 1
              masked wavelengths may have x>1, x<-1
        """
        x = wavelength - (wavelength[mask]).min()
        x = 2.0 * (x / (x[mask]).max()) - 1.0
        return x

    def spec_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        pars = ['gp_jitter', 'gp_amplitude', 'gp_length']
        defaults = [[0.0], [0.0], [1.0]]
        vals = [self.params.get(p, d) for p, d in zip(pars, defaults)]
        return tuple(vals)

    def phot_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        s = self.params.get('phot_jitter', 0.0)
        return s, [0.0], [0]


class PolySedModel(SedModel):
    """This is a subclass of SedModel that replaces the calibration vector with
    the maximum likelihood chebyshev polynomial describing the difference
    between the observed and the model spectrum.
    """

    def spec_calibration(self, theta=None, obs=None, **kwargs):
        """Implements a Chebyshev polynomial calibration model. This uses
        least-squares to find the maximum-likelihood Chebyshev polynomial of a
        certain order describing the ratio of the observed spectrum to the
        model spectrum, conditional on all other parameters, using least
        squares.  The first coefficient is always set to 1, as the overall
        normalization is controlled by ``spec_norm``.

        :returns cal:
           A polynomial given by 'spec_norm' * (1 + \Sum_{m=1}^M
           a_{m} * T_m(x)).
        """
        if theta is not None:
            self.set_parameters(theta)

        norm = self.params.get('spec_norm', 1.0)
        polyopt = ((self.params.get('polyorder', 0) > 0) &
                   (obs.get('spectrum', None) is not None)) 
        if polyopt:
            order = self.params['polyorder']
            mask = obs.get('mask', slice(None))
            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = self.wave_to_x(obs["wavelength"], mask)
            y = (obs['spectrum'] / self._spec)[mask] / norm - 1.0
            yerr = (obs['unc'] / self._spec)[mask] / norm
            yvar = yerr**2
            A = chebvander(x[mask], order)[:, 1:]
            ATA = np.dot(A.T, A / yvar[:, None])
            reg = self.params.get('poly_regularization', 0.)
            if np.any(reg > 0):
                ATA += reg**2 * np.eye(order)
            ATAinv = np.linalg.inv(ATA)
            c = np.dot(ATAinv, np.dot(A.T, y / yvar))
            Afull = chebvander(x, order)[:, 1:]
            poly = np.dot(Afull, c)
            self._poly_coeffs = c
        else:
            poly = 0.0

        return (1.0 + poly) * norm


def gauss(x, mu, A, sigma):
    """Sample multiple gaussians at positions x.

    :param x:
        locations where samples are desired.

    :param mu:
        Center(s) of the gaussians.

    :param A:
        Amplitude(s) of the gaussians, defined in terms of total area.

    :param sigma:
        Dispersion(s) of the gaussians, un units of x.

    :returns val:
        The values of the sum of gaussians at x.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A / (sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2))
    return val.sum(axis=-1)
