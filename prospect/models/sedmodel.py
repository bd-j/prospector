import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander
from .parameters import ProspectorParams
from scipy.stats import multivariate_normal as mvn

from sedpy.observate import getSED

from ..sources.constants import to_cgs_at_10pc as to_cgs
from ..sources.constants import cosmo, lightspeed, ckms, jansky_cgs
from ..utils.smoothing import smoothspec


__all__ = ["SedModel", "SpecModel", "PolySedModel"]


class SedModel(ProspectorParams):
    """A subclass of :py:class:`ProspectorParams` that passes the models
    through to an ``sps`` object and returns spectra and photometry, including
    optional spectroscopic calibration and sky emission.
    """

    def mean_model(self, theta, obs, sps=None, sigma=None, **extras):
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

        :param sigma:
            The covariance matrix for the spectral noise. It is only used for 
            emission line marginalization.

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
            tiny = 1.0 / len(spec) * spec[spec > 0].min()
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


class SpecModel(ProspectorParams):
    """A subclass of :py:class:`ProspectorParams` that passes the models
    through to an ``sps`` object and returns spectra and photometry, including
    optional spectroscopic calibration and sky emission.
    """

    def predict(self, theta, obs, sps=None, sigma_spec=None, **extras):

        # generate and cache model spectrum and info
        self.set_parameters(theta)
        self._wave, self._spec, self._mfrac = sps.get_galaxy_spectrum(**self.params)
        self._zred = self.params.get('zred', 0)
        self._eline_wave, self._eline_lum = sps.get_galaxy_elines()

        # Flux normalize
        self._norm_spec = self._spec * self.flux_norm()

        # generate spectrum and photometry for likelihood
        # predict_spec should be called before predict_phot
        spec = self.predict_spec(obs, sigma_spec)
        phot = self.predict_phot(obs['filters'])

        return spec, phot, self._mfrac

    def predict_spec(self, obs, sigma_spec, **extras):

        # redshift wavelength
        obs_wave = self.observed_wave(self._wave, do_wavecal=True)
        self._outwave = obs.get('wavelength', obs_wave)

        # cache eline parameters
        self.cache_eline_parameters()

        # smooth and put on output wavelength grid
        smooth_spec = self.smoothspec(obs_wave, self._norm_spec)

        # calibration
        self._speccal = self.spec_calibration(obs=obs, spec=smooth_spec, **extras)
        calibrated_spec = smooth_spec * self._speccal

        # generate (after fitting) the emission line spectrum
        emask = self._eline_wavelength_mask
        if self.params.get('marginalize_elines', False):
            self._elinespec = self.get_el(obs, calibrated_spec, sigma_spec)
        elif self.params.get("nebemlineinspec", True):
            self._elinespec = np.zeros_like(emask,dtype=float)[:,None]
        else:
            self._elinespec = self.get_eline_spec()
        calibrated_spec[emask] += self._elinespec.sum(axis=1)

        return calibrated_spec

    def predict_phot(self, filters):

        if filters is None:
            return 0.0

        # generate photometry w/o emission lines
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        flambda = self._norm_spec * lightspeed / obs_wave**2 * (3631*jansky_cgs)
        mags = getSED(obs_wave, flambda, filters)
        phot = np.atleast_1d(10**(-0.4 * mags))

        # generate emission-line photometry
        phot += self.nebline_photometry(filters)

        return phot

    def nebline_photometry(self, filters):
        """analytically calculate emission line contribution to photometry
            fixme: check units of emission line luminosities
        """
        elams = self._ewave_obs
        elums = self._eline_lum * self.flux_norm() / (1 + self._zred)

        flux = np.zeros(len(filters))
        for i, filt in enumerate(filters):
            # calculate transmission at nebular emission
            trans = np.interp(elams, filt.wavelength, filt.transmission,
                              left=0., right=0.)
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*elams[idx]*elums[idx]).sum() / filt.ab_zero_counts

        return flux

    def flux_norm(self):
        """Compute the scaling required to go from Lsun/Hz/Msun to maggies.
        Note this includes the (1+z) factor required for flux densities.
        """
        # distance factor
        if (self._zred == 0) | ('lumdist' in self.params):
            lumdist = self.params.get('lumdist', 1e-5)
        else:
            lumdist = cosmo.luminosity_distance(self._zred).to('Mpc').value
        dfactor = (lumdist * 1e5)**2

        # Mass normalization
        mass = np.sum(self.params.get('mass',1.0))

        # units
        unit_conversion = to_cgs / (3631*jansky_cgs) * (1 + self._zred)

        # this is a scalar
        return mass * unit_conversion / dfactor

    def cache_eline_parameters(self):
        """ This computes and caches:

        * _ewave_obs - The observed frame wavelengths (AA) of all emission lines
        * _eline_sigma_lambda - The dispersion (in AA) of all the emission lines
        * _elines_to_fit - If fitting and marginalizing over emission lines, this
                           stores indices of the lines to actually fit, as a
                           boolean array

        Can be subclassed to add more sophistication
        redshift: first looks for ``eline_z``, and defaults to ``zred``
        sigma: first looks for ``eline_sigma``, defaults to 100 km/s
        """

        # observed wavelengths
        eline_z = self.params.get("eline_delta_zred", 0.0)
        self._ewave_obs = (1 + eline_z + self._zred) * self._eline_wave

        # observed linewidths
        nline = self._ewave_obs.shape[0]
        self._eline_sigma_kms = np.atleast_1d(self.params.get('eline_sigma', 100.0))
        self._eline_sigma_kms = (self._eline_sigma_kms[None] * np.ones(nline)).squeeze()
        #self._eline_sigma_lambda = eline_sigma_kms * self._ewave_obs / ckms

        # --- lines to fit ---
        # lines specified by user, but remove any lines whose central
        # wavelengths are outside the observed spectral range
        elines_index = self.params.get('lines_to_fit',slice(None))
        wmin, wmax = self._outwave.min(), self._outwave.max()
        in_range = (self._ewave_obs.squeeze() > wmin) & (self._ewave_obs.squeeze() < wmax)
        self._elines_to_fit = in_range & elines_index

        # --- wavelengths corresponding to those lines ---
        # within N sigma of the central wavelength
        nsigma = 4
        idx = self._elines_to_fit
        ewave_obs = self._ewave_obs[idx]
        eline_sigma_lambda = self._ewave_obs[idx] / ckms * self._eline_sigma_kms[idx]
        new_mask = np.abs(self._outwave-ewave_obs[:,None]) < nsigma*eline_sigma_lambda[:,None]
        self._eline_wavelength_mask = new_mask.any(axis=0)

    def get_eline_gaussians(self, lineidx=slice(None), wave=None):
        """Get a set of unit normals with centers and widths given by the
        previously cached emission line observed frame wavelengths and emission
        line widths.

        :param lineidx: (optional)
            A boolean array or integer array used to subscript the cached
            lines.  Gaussian vectors will only be constructed for the lines
            thus subscripted.

        :param wave: (optional)
            The wavelength array (in Angstroms) used to construct the gaussian
            vectors. If not given, the cached `_outwave` array will be used.

        :returns gaussians:
            The unit gaussians for each line, in units Lsun/Hz.  ndarray of shape (nwave, nline)
        """
        if wave is None:
            warr = self._outwave
        else:
            warr = wave

        # generate gaussians
        mu = np.atleast_2d(self._ewave_obs[lineidx])
        sigma = np.atleast_2d(self._eline_sigma_kms[lineidx])
        dv = ckms * (warr[:, None]/mu - 1)
        dv_dnu = ckms * warr[:,None]**2 / (lightspeed * mu)

        eline_gaussians = 1. / (sigma * np.sqrt(np.pi * 2)) * np.exp(-dv**2 / (2 * sigma**2))
        eline_gaussians *= dv_dnu

        return eline_gaussians

    def get_el(self, obs, calibrated_spec, sigma_spec=None):
        """ checkme: time the gaussian generation
        if slow, consider alternatives (unit gaussian, interpolate to wavelength grid?)
        """

        # ensure we have no emission lines in spectrum
        # and we definitely want them.
        assert self.params['nebemlineinspec'] == False
        assert self.params['add_neb_emission'] == True

        # generate Gaussians on appropriate wavelength gride
        idx = self._elines_to_fit
        emask = self._eline_wavelength_mask
        nebwave = self._outwave[emask]
        eline_gaussians = self.get_eline_gaussians(lineidx=idx,wave=nebwave)

        # generate residuals
        delta = obs['spectrum'][emask] - calibrated_spec[emask]

        # generate line amplitudes in observed flux units
        units_factor = self.flux_norm() / (1 + self._zred)
        calib_factor = np.interp(self._ewave_obs[idx], nebwave, self._speccal[emask])
        linecal = units_factor * calib_factor
        alpha_breve = self._eline_lum[idx] * linecal

        # generate inverse of sigma_spec
        if sigma_spec is None:
            sigma_spec = obs["unc"]**2
        sigma_spec = sigma_spec[emask]
        if sigma_spec.ndim == 2:
            sigma_inv = np.linalg.inv(sigma_spec)
        else: 
            sigma_inv = np.diag(1. / sigma_spec)

        # calculate emission line amplitudes and covariance matrix
        sigma_alpha_hat = np.linalg.inv(np.dot(eline_gaussians.T, np.dot(sigma_inv, eline_gaussians)))
        alpha_hat = np.dot(sigma_alpha_hat, np.dot(eline_gaussians.T, np.dot(sigma_inv, delta)))

        # generate likelihood penalty term
        # different if we use a prior
        # FIXME: Cache line amplitude covariance matrices
        if self.params.get('use_eline_prior', False):
            sigma_alpha_breve = np.diag(self.params['eline_prior_width'] * alpha_breve)
            M = np.linalg.inv(sigma_alpha_hat + sigma_alpha_breve)
            alpha_bar = (np.dot(sigma_alpha_breve, np.dot(M, alpha_hat)) +
                         np.dot(sigma_alpha_hat, np.dot(M, alpha_breve)))
            sigma_alpha_bar = np.dot(sigma_alpha_hat, np.dot(M, sigma_alpha_breve))
            K = ln_mvn(alpha_hat, mean=alpha_breve, cov=sigma_alpha_breve) - \
                ln_mvn(alpha_hat, mean=alpha_bar, cov=sigma_alpha_bar)
            #K = mvn.pdf(alpha_hat, mean=alpha_breve, cov=sigma_alpha_breve) / \
            #    mvn.pdf(alpha_hat, mean=alpha_breve, cov=sigma_alpha_hat + sigma_alpha_breve)

        else:
            K = ln_mvn(alpha_hat, mean=alpha_hat, cov=sigma_alpha_hat)
            alpha_bar = alpha_hat

        # Cache the ln-penalty
        self._ln_eline_penalty = K

        # Store fitted emission line luminosities in physical units
        self._eline_lum[idx] = alpha_bar / linecal

        # return the maximum-likelihood line spectrum in observed units
        return alpha_hat * eline_gaussians

    def smoothspec(self, wave, spec):
        sigma = self.params.get("sigma_smooth", 100)
        outspec = smoothspec(wave, spec, sigma, outwave=self._outwave, **self.params)

        return outspec

    def observed_wave(self, wave, do_wavecal=False):
        # missing wavelength calibration (add later)
        a = 1 + self._zred
        return wave * a

    def wave_to_x(self, wavelength=None, mask=slice(None), **extras):
        """Map unmasked wavelengths to the interval -1, 1
              masked wavelengths may have x>1, x<-1
        """
        x = wavelength - (wavelength[mask]).min()
        x = 2.0 * (x / (x[mask]).max()) - 1.0
        return x

    def spec_calibration(self, **kwargs):
        return np.ones_like(self._outwave)

    def get_eline_spec(self):
        """returns model emission line spectrum
        only run after calling predict(), as it accesses cached information
        generates an (Nline,Nwave) array
        relatively slow, useful for display purposes
        """
        wave = self._outwave
        emask = self._eline_wavelength_mask
        gaussians = self.get_eline_gaussians(wave=wave[emask])
        elums = self._eline_lum * self.flux_norm() / (1 + self._zred)
        return elums * gaussians * self._speccal[emask,None]


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


class PolySpecModel(SpecModel):
    """This is a subclass of *SpecModel* that replaces the calibration vector with
    the maximum likelihood chebyshev polynomial describing the difference
    between the observed and the model spectrum.
    """

    def spec_calibration(self, theta=None, obs=None, spec=None, **kwargs):
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

            # generate mask
            # remove region around emission lines if doing analytical marginalization
            mask = obs.get('mask', slice(None))
            if self.params.get('marginalize_elines', False):
                mask[self._eline_wavelength_mask] = 0

            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = self.wave_to_x(obs["wavelength"], mask)
            y = (obs['spectrum'] / spec)[mask] / norm - 1.0
            yerr = (obs['unc'] / spec)[mask] / norm
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
            poly = np.zeros_like(self._outwave)

        return (1.0 + poly) * norm

class PolyFitModel(SedModel):

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

def ln_mvn(x,mean=None,cov=None):
    """  Calculates the natural logarithm of the multivariate normal PDF evaluated at `x`

    :param x:
        locations where samples are desired.

    :param mean:
        Center(s) of the gaussians.

    :param cov:
        Covariances of the gaussians.

    """

    ndim = mean.shape[-1]
    dev = x - mean
    log_2pi = np.log( 2 * np.pi)
    log_det = np.log(np.linalg.det(cov))
    exp = np.dot(dev.T,np.dot(np.linalg.inv(cov), dev))

    return -0.5 * (ndim * log_2pi + log_det + exp)

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
