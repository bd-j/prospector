#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sedmodel.py - classes and methods for storing parameters and predicting
observed spectra and photometry from them, given a Source object.
"""

import numpy as np
import os

from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.interpolate import splrep, BSpline
from scipy.stats import multivariate_normal as mvn

from sedpy.observate import getSED

from .parameters import ProspectorParams
from ..sources.constants import to_cgs_at_10pc as to_cgs
from ..sources.constants import cosmo, lightspeed, ckms, jansky_cgs
from ..utils.smoothing import smoothspec


__all__ = ["SpecModel", "PolySpecModel", "SplineSpecModel",
           "SedModel", "PolySedModel", "PolyFitModel"]


class SpecModel(ProspectorParams):

    """A subclass of :py:class:`ProspectorParams` that passes the models
    through to an ``sps`` object and returns spectra and photometry, including
    optional spectroscopic calibration, and sky emission.

    This class performs most of the conversion from intrinsic model spectrum to
    observed quantities, and additionally can compute MAP emission line values
    and penalties for marginalization over emission line amplitudes.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # get the emission line info
        try:
            SPS_HOME = os.getenv('SPS_HOME')
            info = np.genfromtxt(os.path.join(SPS_HOME, 'data', 'emlines_info.dat'),
                                 dtype=[('wave', 'f8'), ('name', '<U20')],
                                 delimiter=',')
            self.emline_info = info
            self._use_eline = np.ones(len(info), dtype=bool)
        except(OSError, KeyError, ValueError) as e:
            print("Could not read and cache emission line info from $SPS_HOME/data/emlines_info.dat")
            self.emline_info = e

    def _available_parameters(self):
        new_pars = [("sigma_smooth", ""),
                    ("marginalize_elines", ""),
                    ("elines_to_fit", ""),
                    ("elines_to_fix", ""),
                    ("elines_to_ignore", ""),
                    ("eline_delta_zred", ""),
                    ("eline_sigma", ""),
                    ("use_eline_priors", ""),
                    ("eline_prior_width", "")]
        relevant_pars = [("mass", ""),
                         ("lumdist", ""),
                         ("zred", ""),
                         ("nebemlineinspec", ""),
                         ("add_neb_emission")]

        return new_pars

    def predict(self, theta, obs=None, sps=None, sigma_spec=None, **extras):
        """Given a ``theta`` vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        :param theta:
            ndarray of parameter values, of shape ``(ndim,)``

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the observed fluxes and
            uncertainties thereon.  Assumed to be the result of
            :py:func:`utils.obsutils.rectify_obs`

        :param sps:
            An `sps` object to be used in the model generation.  It must have
            the :py:func:`get_galaxy_spectrum` method defined.

        :param sigma_spec: (optional)
            The covariance matrix for the spectral noise. It is only used for
            emission line marginalization.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by ``obs['wavelength']``, including multiplication by the
            calibration vector.  Units of maggies

        :returns phot:
            The model photometry for these parameters, for the filters
            specified in ``obs['filters']``.  Units of maggies.

        :returns extras:
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            stellar mass formed.
        """
        # generate and cache model spectrum and info
        self.set_parameters(theta)
        self._wave, self._spec, self._mfrac = sps.get_galaxy_spectrum(**self.params)
        self._zred = self.params.get('zred', 0)
        self._eline_wave, self._eline_lum = sps.get_galaxy_elines()

        # Flux normalize
        self._norm_spec = self._spec * self.flux_norm()

        # generate spectrum and photometry for likelihood
        # predict_spec should be called before predict_phot
        # because in principle it can modify the emission line parameters
        # and also needs some things done in 'cache_eline_parameters`
        # especially _ewave_obs and _use_elines
        spec = self.predict_spec(obs, sigma_spec)
        phot = self.predict_phot(obs.get('filters', None))

        return spec, phot, self._mfrac

    def predict_spec(self, obs, sigma_spec, **extras):
        """Generate a prediction for the observed spectrum.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct

          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies
          + ``_eline_wave`` and ``_eline_lum`` - emission line parameters from the SPS model

        It generates the following attributes

          + ``_outwave`` - Wavelength grid (observed frame)
          + ``_speccal`` - Calibration vector
          + ``_sed`` - Intrinsic spectrum (before cilbration vector applied but including emission lines)

        And the following attributes are generated if nebular lines are added

          + ``_fix_eline_spec`` - emission line spectrum for fixed lines, intrinsic units
          + ``_fix_eline_spec`` - emission line spectrum for fitted lines, with
            spectroscopic calibration factor included.

        Numerous quantities related to the emission lines are also cached (see
        ``cache_eline_parameters()`` and ``get_el()`` for details.)

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the observed fluxes and
            uncertainties thereon.  Assumed to be the result of
            :py:meth:`utils.obsutils.rectify_obs`

        :param sigma_spec: (optional)
            The covariance matrix for the spectral noise. It is only used for
            emission line marginalization.

        :returns spec:
            The prediction for the observed frame spectral flux these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            including multiplication by the calibration vector.
            ndarray of shape ``(nwave,)`` in units of maggies.
        """
        # redshift wavelength
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        self._outwave = obs.get('wavelength', obs_wave)
        if self._outwave is None:
            self._outwave = obs_wave

        # --- cache eline parameters ---
        self.cache_eline_parameters(obs)

        # --- smooth and put on output wavelength grid ---
        smooth_spec = self.smoothspec(obs_wave, self._norm_spec)

        # --- add fixed lines if necessary ---
        emask = self._fix_eline_pixelmask
        if emask.any():
            inds = self._fix_eline & self._valid_eline
            espec = self.predict_eline_spec(line_indices=inds,
                                            wave=self._outwave[emask])
            self._fix_eline_spec = espec
            smooth_spec[emask] += self._fix_eline_spec.sum(axis=1)

        # --- calibration ---
        self._speccal = self.spec_calibration(obs=obs, spec=smooth_spec, **extras)
        calibrated_spec = smooth_spec * self._speccal

        # --- fit and add lines if necessary ---
        emask = self._fit_eline_pixelmask
        if emask.any():
            self._fit_eline_spec = self.get_el(obs, calibrated_spec, sigma_spec)
            calibrated_spec[emask] += self._fit_eline_spec.sum(axis=1)

        # --- cache intrinsic spectrum ---
        self._sed = calibrated_spec / self._speccal

        return calibrated_spec

    def predict_phot(self, filters):
        """Generate a prediction for the observed photometry.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct:
          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies.
          + ``_ewave_obs`` and ``_eline_lum`` - emission line parameters from
            the SPS model

        :param filters:
            Instance of :py:class:`sedpy.observate.FilterSet` or list of
            :py:class:`sedpy.observate.Filter` objects. If there is no
            photometry, ``None`` should be supplied.

        :returns phot:
            Observed frame photometry of the model SED through the given filters.
            ndarray of shape ``(len(filters),)``, in units of maggies.
            If ``filters`` is None, this returns 0.0
        """
        if filters is None:
            return 0.0

        # generate photometry w/o emission lines
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        flambda = self._norm_spec * lightspeed / obs_wave**2 * (3631*jansky_cgs)
        phot = 10**(-0.4 * np.atleast_1d(getSED(obs_wave, flambda, filters)))
        # TODO: below is faster for sedpy > 0.2.0
        #phot = np.atleast_1d(getSED(obs_wave, flambda, filters, linear_flux=True))

        # generate emission-line photometry
        if (self._want_lines & self._need_lines):
            phot += self.nebline_photometry(filters)

        return phot

    def flux_norm(self):
        """Compute the scaling required to go from Lsun/Hz/Msun to maggies.
        Note this includes the (1+z) factor required for flux densities.

        :returns norm: (float)
            The normalization factor, scalar float.
        """
        # distance factor
        if (self._zred == 0) | ('lumdist' in self.params):
            lumdist = self.params.get('lumdist', 1e-5)
        else:
            lumdist = cosmo.luminosity_distance(self._zred).to('Mpc').value
        dfactor = (lumdist * 1e5)**2
        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        # units
        unit_conversion = to_cgs / (3631*jansky_cgs) * (1 + self._zred)

        return mass * unit_conversion / dfactor

    @property
    def _need_lines(self):
        return (not (bool(self.params.get("nebemlineinspec", True))))

    @property
    def _want_lines(self):
        return bool(self.params.get('add_neb_emission', False))

    def nebline_photometry(self, filters, elams=None, elums=None):
        """Compute the emission line contribution to photometry.  This requires
        several cached attributes:
          + ``_ewave_obs``
          + ``_eline_lum``

        :param filters:
            Instance of :py:class:`sedpy.observate.FilterSet` or list of
            :py:class:`sedpy.observate.Filter` objects

        :param elams: (optional)
            The emission line wavelength in angstroms.  If not supplied uses the
            cached ``_ewave_obs`` attribute.

        :param elums: (optional)
            The emission line flux in erg/s/cm^2.  If not supplied uses  the
            cached ``_eline_lum`` attribute and applies appropriate distance
            dimming and unit conversion.

        :returns nebflux:
            The flux of the emission line through the filters, in units of
            maggies. ndarray of shape ``(len(filters),)``
        """
        if (elams is None) or (elums is None):
            elams = self._ewave_obs[self._use_eline]
            # We have to remove the extra (1+z) since this is flux, not a flux density
            # Also we convert to cgs
            self.line_norm = self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs)
            elums = self._eline_lum[self._use_eline] * self.line_norm

        # loop over filters
        flux = np.zeros(len(filters))
        try:
            # TODO: Since in this case filters are on a grid, there should be a
            # faster way to look up the transmission than the later loop
            flist = filters.filters
        except(AttributeError):
            flist = filters
        for i, filt in enumerate(flist):
            # calculate transmission at line wavelengths
            trans = np.interp(elams, filt.wavelength, filt.transmission,
                              left=0., right=0.)
            # include all lines where transmission is non-zero
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*elams[idx]*elums[idx]).sum() / filt.ab_zero_counts

        return flux

    def cache_eline_parameters(self, obs, nsigma=5):
        """ This computes and caches a number of quantities that are relevant
        for predicting the emission lines, and computing the MAP values thereof,
        including
          + ``_ewave_obs`` - Observed frame wavelengths (AA) of all emission lines.
          + ``_eline_sigma_kms`` - Dispersion (in km/s) of all the emission lines
          + ``_fit_eline`` - If fitting and marginalizing over emission lines,
            this stores a boolean mask of the lines to actually fit. Only lines
            that are within ``nsigma`` of an observed wavelength points are
            included.
          + ``_fix_eline`` - this stores a boolean mask of the lines that are
            to be added with the cloudy amplitudes Only lines that are within
            ``nsigma`` of an observed wavelength point are included.
          + ``_fit_eline_pixelmask`` - A mask of the `_outwave` vector that
            indicates which pixels to use in the emission line fitting.
            Only pixels within ``nsigma`` of an emission line are used.
          + ``_fix_eline_pixelmask`` - A mask of the `_outwave` vector that
            indicates which pixels to use in the fixed emission line prediction.

        Can be subclassed to add more sophistication
        redshift - first looks for ``eline_delta_zred``, and defaults to ``zred``
        sigma - first looks for ``eline_sigma``, defaults to 100 km/s

        :param nsigma: (float, optional, default: 5.)
            Number of sigma from a line center to use for defining which lines
            to fit and useful spectral elements for the fitting.  float.
        """

        # observed wavelengths
        eline_z = self.params.get("eline_delta_zred", 0.0)
        self._ewave_obs = (1 + eline_z + self._zred) * self._eline_wave

        # masks for lines to be treated in various ways.
        # always run this becuase it's need for spec *and* phot if adding lines
        # by hand
        self.parse_elines()

        # exit gracefully if not adding lines.  We also exit if only fitting
        # photometry, for performance reasons
        hasspec = obs.get('spectrum', None) is not None
        if not (self._want_lines & self._need_lines & hasspec):
            self._fit_eline = None
            self._fit_eline_pixelmask = np.array([], dtype=bool)
            self._fix_eline_pixelmask = np.array([], dtype=bool)
            self._fix_eline = None
            return

        # observed linewidths
        nline = self._ewave_obs.shape[0]
        self._eline_sigma_kms = np.atleast_1d(self.params.get('eline_sigma', 100.0))
        # what is this wierd construction for?
        self._eline_sigma_kms = (self._eline_sigma_kms[None] * np.ones(nline)).squeeze()
        #self._eline_sigma_lambda = eline_sigma_kms * self._ewave_obs / ckms

        # --- get valid lines ---
        # fixed and fit lines specified by user, but remove any lines which do
        # not have an observed pixel within 5sigma of their center
        # This part has to go in every call
        linewidth = nsigma * self._ewave_obs / ckms * self._eline_sigma_kms
        pixel_mask = (np.abs(self._outwave - self._ewave_obs[:, None]) < linewidth[:, None])
        self._valid_eline = pixel_mask.any(axis=1) & self._use_eline

        # --- wavelengths corresponding to valid lines ---
        # within N sigma of the central wavelength
        self._fit_eline_pixelmask = pixel_mask[self._valid_eline & self._fit_eline, :].any(axis=0)
        self._fix_eline_pixelmask = pixel_mask[self._valid_eline & self._fix_eline, :].any(axis=0)
        # --- lines to fit ---
        self._elines_to_fit = self._fit_eline & self._valid_eline

    def parse_elines(self):
        """Create mask arrays to identify the lines that should be fit and the
        lines that should be fixed based on the content of `params["lines_to_fix"]`
        and `params["lines_to_fit"]`

        This can probably be cached just once unless you want to change between
        separate fits.
        """

        all_lines = self.emline_info['name']

        if self.params.get('marginalize_elines', False):
            # if marginalizing, default to fitting all lines
            # unless some are explicitly fixed
            lnames_to_fit = self.params.get('elines_to_fit', all_lines)
            lnames_to_fix = self.params.get('elines_to_fix', np.array([]))
            self._fit_eline = np.isin(all_lines, lnames_to_fit) & ~np.isin(all_lines, lnames_to_fix)
        else:
            self._fit_eline = np.zeros(len(all_lines), dtype=bool)

        self._fix_eline = ~self._fit_eline

        if self.params.get("elines_to_ignore", []):
            self._use_eline = ~np.isin(self.emline_info["name"],
                                       self.params["elines_to_ignore"])

    def get_el(self, obs, calibrated_spec, sigma_spec=None):
        """Compute the maximum likelihood and, optionally, MAP emission line
        amplitudes for lines that fall within the observed spectral range. Also
        compute and cache the analytic penalty to log-likelihood from
        marginalizing over the emission line amplitudes.  This is cached as
        ``_ln_eline_penalty``.  The emission line amplitudes (in maggies) at
        `_eline_lums` are updated to the ML values for the fitted lines.

        :param obs:
            A dictionary containing the ``'spectrum'`` and ``'unc'`` keys that
            are observed fluxes and uncertainties, both ndarrays of shape
            ``(n_wave,)``

        :param calibrated_spec:
            The predicted (so far) observer-frame spectrum in the same units as
            the observed spectrum, ndarray of shape ``(n_wave,)``  Should
            include pixed lines but not lines to be fit

        :param sigma_spec:
            Spectral covariance matrix, if using a non-trivial noise model.

        :returns el:
            The maximum likelihood emission line flux densities.
            ndarray of shape ``(n_wave_neb, n_fitted_lines)`` where
            ``n_wave_neb`` is the number of wavelength elements within
            ``nsigma`` of a line, and ``n_fitted_lines`` is the number of lines
            that fall within ``nsigma`` of a wavelength pixel.  Units are same
            as ``calibrated_spec``
        """
        # ensure we have no emission lines in spectrum
        # and we definitely want them.
        assert bool(self.params['nebemlineinspec']) is False
        assert bool(self.params['add_neb_emission']) is True

        # generate Gaussians on appropriate wavelength grid
        idx = self._elines_to_fit
        emask = self._fit_eline_pixelmask
        nebwave = self._outwave[emask]
        eline_gaussians = self.get_eline_gaussians(lineidx=idx, wave=nebwave)

        # generate residuals
        delta = obs['spectrum'][emask] - calibrated_spec[emask]

        # generate line amplitudes in observed flux units
        units_factor = self.flux_norm() / (1 + self._zred)
        calib_factor = np.interp(self._ewave_obs[idx], nebwave, self._speccal[emask])
        linecal = units_factor * calib_factor
        alpha_breve = self._eline_lum[idx] * linecal

        # FIXME: nebopt: be careful with inverses
        # generate inverse of sigma_spec
        if sigma_spec is None:
            sigma_spec = obs["unc"]**2
        sigma_spec = sigma_spec[emask]
        if sigma_spec.ndim == 2:
            sigma_inv = np.linalg.pinv(sigma_spec)
        else:
            sigma_inv = np.diag(1. / sigma_spec)

        # calculate ML emission line amplitudes and covariance matrix
        # FIXME: nebopt: do this with a solve
        sigma_alpha_hat = np.linalg.pinv(np.dot(eline_gaussians.T, np.dot(sigma_inv, eline_gaussians)))
        alpha_hat = np.dot(sigma_alpha_hat, np.dot(eline_gaussians.T, np.dot(sigma_inv, delta)))

        # generate likelihood penalty term (and MAP amplitudes)
        # FIXME: Cache line amplitude covariance matrices?
        if self.params.get('use_eline_prior', False):
            # Incorporate gaussian priors on the amplitudes
            sigma_alpha_breve = np.diag((self.params['eline_prior_width'] * np.abs(alpha_breve)))**2
            M = np.linalg.pinv(sigma_alpha_hat + sigma_alpha_breve)
            alpha_bar = (np.dot(sigma_alpha_breve, np.dot(M, alpha_hat)) +
                         np.dot(sigma_alpha_hat, np.dot(M, alpha_breve)))
            sigma_alpha_bar = np.dot(sigma_alpha_hat, np.dot(M, sigma_alpha_breve))
            # FIXME: nebopt: can we avoid a scipy call here
            K = ln_mvn(alpha_hat, mean=alpha_breve, cov=sigma_alpha_breve+sigma_alpha_hat) - \
                ln_mvn(alpha_hat, mean=alpha_hat, cov=sigma_alpha_hat)
        else:
            # simply use the ML values and associated marginaliztion penalty
            alpha_bar = alpha_hat
            K = ln_mvn(alpha_hat, mean=alpha_hat, cov=sigma_alpha_hat)

        # Cache the ln-penalty
        self._ln_eline_penalty = K

        # Store fitted emission line luminosities in physical units
        self._eline_lum[idx] = alpha_bar / linecal

        # return the maximum-likelihood line spectrum in observed units
        return alpha_hat * eline_gaussians

    def predict_eline_spec(self, line_indices=slice(None), wave=None):
        """Compute a complete model emission line spectrum. This should only
        be run after calling predict(), as it accesses cached information.
        Relatively slow, useful for display purposes

        :param line_indices: optional
            If given, this should give the indices of the lines to predict.

        :param wave: (optional, default: ``None``)
            The wavelength ndarray on which to compute the emission line spectrum.
            If not supplied, the ``_outwave`` vector is used.

        :returns eline_spec:
            An (n_line, n_wave) ndarray, units of Lsun/Hz intrinsic (no
            calibration vector applied)
        """
        gaussians = self.get_eline_gaussians(lineidx=line_indices, wave=wave)
        elums = self._eline_lum[line_indices] * self.flux_norm() / (1 + self._zred)
        return elums * gaussians

    def get_eline_gaussians(self, lineidx=slice(None), wave=None):
        """Generate a set of unit normals with centers and widths given by the
        previously cached emission line observed-frame wavelengths and emission
        line widths.

        :param lineidx: (optional)
            A boolean array or integer array used to subscript the cached
            lines.  Gaussian vectors will only be constructed for the lines
            thus subscripted.

        :param wave: (optional)
            The wavelength array (in Angstroms) used to construct the gaussian
            vectors. If not given, the cached `_outwave` array will be used.

        :returns gaussians:
            The unit gaussians for each line, in units Lsun/Hz.
            ndarray of shape (n_wave, n_line)
        """
        if wave is None:
            warr = self._outwave
        else:
            warr = wave

        # generate gaussians
        mu = np.atleast_2d(self._ewave_obs[lineidx])
        sigma = np.atleast_2d(self._eline_sigma_kms[lineidx])
        dv = ckms * (warr[:, None]/mu - 1)
        dv_dnu = ckms * warr[:, None]**2 / (lightspeed * mu)

        eline_gaussians = 1. / (sigma * np.sqrt(np.pi * 2)) * np.exp(-dv**2 / (2 * sigma**2))
        eline_gaussians *= dv_dnu

        # outside of the wavelengths defined by the spectrum? (why this dependence?)
        # FIXME what is this?
        eline_gaussians /= -np.trapz(eline_gaussians, 3e18/warr[:, None], axis=0)

        return eline_gaussians

    def smoothspec(self, wave, spec):
        """Smooth the spectrum.  See :py:func:`prospect.utils.smoothing.smoothspec`
        for details.
        """
        sigma = self.params.get("sigma_smooth", 100)
        outspec = smoothspec(wave, spec, sigma, outwave=self._outwave, **self.params)

        return outspec

    def observed_wave(self, wave, do_wavecal=False):
        """Convert the restframe wavelngth grid to the observed frame wavelength
        grid, optionally including wavelength calibration adjustments.  Requires
        that the ``_zred`` attribute is already set.

        :param wave:
            The wavelength array
        """
        # FIXME: missing wavelength calibration code
        if do_wavecal:
            raise NotImplementedError
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

    def absolute_rest_maggies(self, filters):
        """Return absolute rest-frame maggies (=10**(-0.4*M)) of the last
        computed spectrum.

        Parameters
        ----------
        filters : list of ``sedpy.observate.Filter()`` instances
            The filters through which you wish to compute the absolute mags

        Returns
        -------
        maggies : ndarray of shape (nbands,)
            The absolute restframe maggies of the model through the supplied
            filters, including emission lines.  Convert to absolute rest-frame
            magnitudes as M = -2.5 * log10(maggies)
        """
        # --- convert spectrum ---
        ld = cosmo.luminosity_distance(self._zred).to("pc").value
        # convert to maggies if the source was at 10 parsec, accounting for the (1+z) applied during predict()
        fmaggies = self._norm_spec / (1 + self._zred) * (ld / 10)**2
        # convert to erg/s/cm^2/AA for sedpy and get absolute magnitudes
        flambda = fmaggies * lightspeed / self._wave**2 * (3631*jansky_cgs)
        abs_rest_maggies = 10**(-0.4 * np.atleast_1d(getSED(self._wave, flambda, filters)))
        # TODO: below is faster for sedpy > 0.2.0
        #abs_rest_maggies = np.atleast_1d(getSED(self._wave, flambda, filters, linear_flux=True))

        # add emission lines
        if bool(self.params.get('nebemlineinspec', False)) is False:
            eline_z = self.params.get("eline_delta_zred", 0.0)
            elams = (1 + eline_z) * self._eline_wave
            elums = self._eline_lum * self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs) * (ld / 10)**2
            emaggies = self.nebline_photometry(filters, elams=elams, elums=elums)
            abs_rest_maggies += emaggies

        return abs_rest_maggies

    def mean_model(self, theta, obs, sps=None, sigma=None, **extras):
        """Legacy wrapper around predict()
        """
        return self.predict(theta, obs, sps=sps, sigma_spec=sigma, **extras)


class PolySpecModel(SpecModel):

    """This is a subclass of *SpecModel* that generates the multiplicative
    calibration vector at each model `predict` call as the maximum likelihood
    chebyshev polynomial describing the ratio between the observed and the model
    spectrum.
    """

    def _available_parameters(self):
        pars = [("polyorder", "order of the polynomial to fit"),
                ("poly_regularization", "vector of length `polyorder` providing regularization for each polynomial term")
                ]

        return pars

    def spec_calibration(self, theta=None, obs=None, spec=None, **kwargs):
        """Implements a Chebyshev polynomial calibration model. This uses
        least-squares to find the maximum-likelihood Chebyshev polynomial of a
        certain order describing the ratio of the observed spectrum to the model
        spectrum, conditional on all other parameters, using least squares. If
        emission lines are being marginalized out, they are excluded from the
        least-squares fit.

        :returns cal:
           A polynomial given by :math:`\sum_{m=0}^M a_{m} * T_m(x)`.
        """
        if theta is not None:
            self.set_parameters(theta)

        # norm = self.params.get('spec_norm', 1.0)
        polyopt = ((self.params.get('polyorder', 0) > 0) &
                   (obs.get('spectrum', None) is not None))
        if polyopt:
            order = self.params['polyorder']

            # generate mask
            # remove region around emission lines if doing analytical marginalization
            mask = obs.get('mask', np.ones_like(obs['wavelength'], dtype=bool)).copy()
            if self.params.get('marginalize_elines', False):
                mask[self._fit_eline_pixelmask] = 0

            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = self.wave_to_x(obs["wavelength"], mask)
            y = (obs['spectrum'] / spec)[mask] - 1.0
            yerr = (obs['unc'] / spec)[mask]
            yvar = yerr**2
            A = chebvander(x[mask], order)
            ATA = np.dot(A.T, A / yvar[:, None])
            reg = self.params.get('poly_regularization', 0.)
            if np.any(reg > 0):
                ATA += reg**2 * np.eye(order)
            ATAinv = np.linalg.inv(ATA)
            c = np.dot(ATAinv, np.dot(A.T, y / yvar))
            Afull = chebvander(x, order)
            poly = np.dot(Afull, c)
            self._poly_coeffs = c
        else:
            poly = np.zeros_like(self._outwave)

        return (1.0 + poly)


class SplineSpecModel(SpecModel):

    """This is a subclass of *SpecModel* that generates the multiplicative
    calibration vector at each model `predict` call as the maximum likelihood
    cubic spline describing the ratio between the observed and the model
    spectrum.
    """

    def _available_parameters(self):
        pars = [("spline_knot_wave", "vector of wavelengths for the location ot he spline knots"),
                ("spline_knot_spacing", "spacing between knots, in units of wavelength"),
                ("spline_knot_n", "number of interior knots between minimum and maximum unmasked wavelength")
                ]

        return pars

    def spec_calibration(self, theta=None, obs=None, spec=None, **kwargs):
        """Implements a spline calibration model.  This fits a cubic spline with
        determined knot locations to the ratio of the observed spectrum to the
        model spectrum.  If emission lines are being marginalized out, they are
        excluded from the least-squares fit.

        The knot locations must be specified as model parameters, either
        explicitly or via a number of knots or knot spacing (in angstroms)
        """
        if theta is not None:
            self.set_parameters(theta)

        splineopt = True
        if splineopt:
            mask, (wlo, whi) = self.obs_to_mask(obs)
            y = (obs['spectrum'] / spec)[mask] - 1.0
            yerr = (obs['unc'] / spec)[mask]  # HACK
            knots_x = self.make_knots(wlo, whi, as_x=True, **self.params)
            x = 2.0 * (obs["wavelength"] - wlo) / (whi - wlo) - 1.0
            tck = splrep(x[mask], y[mask], w=1/yerr[mask], k=3, task=-1, t=knots_x)
            self._spline_coeffs = tck
            spl = BSpline(*tck)
            spline = spl(x)
        else:
            spline = np.zeros_like(self._outwave)
        return (1.0 + spline)

    def make_knots(self, wlo, whi, as_x=True, **params):
        """
        """
        if "spline_knot_wave" in params:
            knots = np.squeeze(params["spline_knot_wave"])
        elif "spline_knot_spacing" in params:
            s = np.squeeze(params["spline_knot_spacing"])
            # we drop the start so knots are interior
            knots = np.arange(wlo, whi, s)[1:]
        elif "spline_knot_n" in params:
            n = np.squeeze(params["spline_knot_n"])
            # we need to drop the endpoints so knots are interior
            knots = np.linspace(wlo, whi, n)[1:-1]
        else:
            raise KeyError("No valid spline knot specification in self.params")

        if as_x:
            knots = 2.0 * (knots - wlo) / (whi - wlo) - 1.0

        return knots

    def obs_to_mask(self, obs):
        mask = obs.get('mask', np.ones_like(obs['wavelength'], dtype=bool)).copy()
        if self.params.get('marginalize_elines', False):
            mask[self._fit_eline_pixelmask] = 0
        w = obs["wavelength"]
        wrange = w[mask].min(), w[mask].max()
        return mask, wrange


class SedModel(ProspectorParams):

    """A subclass of :py:class:`ProspectorParams` that passes the models
    through to an ``sps`` object and returns spectra and photometry, including
    optional spectroscopic calibration and sky emission.
    """

    def predict(self, theta, obs=None, sps=None, **extras):
        """Given a ``theta`` vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        :param theta:
            ndarray of parameter values, of shape ``(ndim,)``

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the observed fluxes and
            uncertainties thereon.  Assumed to be the result of
            :py:func:`utils.obsutils.rectify_obs`

        :param sps:
            An `sps` object to be used in the model generation.  It must have
            the :py:func:`get_spectrum` method defined.

        :param sigma_spec: (optional, unused)
            The covariance matrix for the spectral noise. It is only used for
            emission line marginalization.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by ``obs['wavelength']``, including multiplication by the
            calibration vector.  Units of maggies

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

    def sed(self, theta, obs=None, sps=None, **kwargs):
        """Given a vector of parameters ``theta``, generate a spectrum, photometry,
        and any extras (e.g. surviving mass fraction), ***not** including any
        instrument calibration effects.  The intrinsic spectrum thus produced is
        cached in `_spec` attribute

        :param theta:
            ndarray of parameter values.

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the observed fluxes and
            uncertainties thereon.  Assumed to be the result of
            :py:func:`utils.obsutils.rectify_obs`

        :param sps:
            An `sps` object to be used in the model generation.  It must have
            the :py:func:`get_spectrum` method defined.

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
        """Implements an overall scaling of the spectrum, given by the
        parameter ``'spec_norm'``

        :returns cal: (float)
          A scalar multiplicative factor that gives the ratio between the true
          spectrum and the observed spectrum
        """
        if theta is not None:
            self.set_parameters(theta)

        return 1.0 * self.params.get('spec_norm', 1.0)

    def wave_to_x(self, wavelength=None, mask=slice(None), **extras):
        """Map unmasked wavelengths to the interval (-1, 1). Masked wavelengths may have x>1, x<-1

        :param wavelength:
            The input wavelengths.  ndarray of shape ``(nwave,)``

        :param mask: optional
            The mask.  slice or boolean array with ``True`` for unmasked elements.
            The interval (-1, 1) will be defined only by unmasked wavelength points

        :returns x:
            The wavelength vector, remapped to the interval (-1, 1).
            ndarray of same shape as  ``wavelength``
        """
        x = wavelength - (wavelength[mask]).min()
        x = 2.0 * (x / (x[mask]).max()) - 1.0
        return x

    def mean_model(self, theta, obs, sps=None, sigma_spec=None, **extras):
        """Legacy wrapper around predict()
        """
        return self.predict(theta, obs, sps=sps, sigma=sigma_spec, **extras)


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
           A polynomial given by 'spec_norm' * (1 + \sum_{m=1}^M a_{m} * T_m(x)).
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


class PolyFitModel(SedModel):

    """This is a subclass of *SedModel* that generates the multiplicative
    calibration vector as a Chebyshev polynomial described by the
    ``'poly_coeffs'`` parameter of the model, which may be free (fittable)
    """

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


def ln_mvn(x, mean=None, cov=None):
    """Calculates the natural logarithm of the multivariate normal PDF
    evaluated at `x`

    :param x:
        locations where samples are desired.

    :param mean:
        Center(s) of the gaussians.

    :param cov:
        Covariances of the gaussians.
    """
    ndim = mean.shape[-1]
    dev = x - mean
    log_2pi = np.log(2 * np.pi)
    sign, log_det = np.linalg.slogdet(cov)
    exp = np.dot(dev.T, np.dot(np.linalg.pinv(cov, rcond=1e-12), dev))

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
