#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sedmodel.py - classes and methods for storing parameters and predicting
observed spectra and photometry from them, given a Source object.
"""

import numpy as np
import os

from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.interpolate import splrep, BSpline
from scipy.signal import medfilt

from sedpy.observate import getSED
from sedpy.smoothing import smoothspec

from .parameters import ProspectorParams
from .hyperparameters import ProspectorHyperParams
from ..sources.constants import to_cgs_at_10pc as to_cgs
from ..sources.constants import cosmo, lightspeed, ckms, jansky_cgs


__all__ = ["SpecModel",
           "HyperSpecModel",
           "AGNSpecModel",
           "AGNPolySpecModel"]


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

        self.init_eline_info()
        self.parse_elines()

    def _available_parameters(self):
        new_pars = [("sigma_smooth", "LOSVD, in km/s, for the stars"),
                    ("marginalize_elines", ""),
                    ("elines_to_fit", ""),
                    ("elines_to_fix", ""),
                    ("elines_to_ignore", ""),
                    ("eline_delta_zred", ""),
                    ("eline_sigma", "LOSVD, in km/s, of nebular emission lines"),
                    ("use_eline_priors", ""),
                    ("eline_prior_width", ""),
                    ("dla_logNh", "log_10 HI column density for damped Lyman-alpha absorption"),
                    ("dla_redshift", "redshift of the DLA; if greater than zred then no absorption occurs"),
                    ("igm_damping", "boolean switch to turn on IGM damping wing redward of 1216 rest")]

        referenced_pars = [("mass", ""),
                           ("lumdist", ""),
                           ("zred", ""),
                           ("nebemlineinspec", ""),
                           ("add_neb_emission")]

        return new_pars

    def quantities(self, obs=None):
        """Return a dictionary of lists of cached quantities generated during
        the last call to :py:func:`predict`.
        """

        # get an emission line spectrum, physically smoothed, on the library grid
        espec = self.predict_eline_spec(line_indices=self.use_eline, wave=self._wave*(1 + self._zred))
        espec_full = espec.sum(axis=-1)

        # Spectra on the restframe library wavelength grid
        l = [(self._wave, "AA", "Library restframe wavelength array"),
             (self._spec, "Lsun/Hz/Msun formed", "Library restframe spectral flux array at library resolution"),
             (self._norm_spec, "maggies", "Observed-frame spectral flux array, at the restframe library wavelength grid and resolution"),
             (self._library_resolution, "km/s", "Spectral resolution of the SPS library"),
             (self._smooth_spec, "maggies", "LOSVD convolved observed-frame spectral flux array, on the library wavelength grid"),
             (espec_full, "maggies", "LOSVD convolved Emission line spectrum (not including nebular continuum) on the library wavelngth grid"),
           ]

        # Spectra on the given observed frame grid
        if obs is not None:
            # make all lines fixed, not fit
            fe = self._fit_eline.copy()
            self._fit_eline = 0 * fe
            self._fix_eline = np.ones_like(self._fix_eline)
            # do the prediction
            pred = self.predict_spec(obs)
            # restore fitted lines
            self._fit_eline = fe # turn the emission line fitting back on
            self._fix_eline = ~fe # turn off the fixed lines when fitting

            # get an emission line spectrum, physically smoothed, on the observed waveelnght grid
            espec = self.predict_eline_spec(line_indices=self.use_eline & self._valid_eline, wave=self._outwave)
            espec_obs = espec.sum(axis=-1)

            o = [(self._outwave, "AA","Observed wavelength array"),
                 (pred, "maggies", "Predicted spectrum on the observed wavelength grid"),
                 (self._sed, "maggies", "Predicted spectrum, before calibration adjustment"),
                 (self._speccal, "observed/intrinsic", "Model flux scaling adjustment ('calibration' or response).  observed = self._sed * self._speccal"),
                 (espec_obs, "maggies", "LOSVD convolved emission line spectrum (not including nebular continuum) on the observed wavelength grid"),
                ]
        else:
            o = None

        # Emission line parameters
        e = [(self._eline_wave, "Restframe wavelengths of nebular emission lines", "AA"),
             (self._eline_lum, "Nebular emission line luminosities", "erg/s"),
             (self._eline_lum_covar, "Covariance matrix of nebular emission line luminosities", "erg^2/s^2"),
            ]

        # Other
        x = [(self._mfrac, "Ratio of surviving stellar mass to formed stellar mass", "dimensionless"),
             (self._zred, "Redshift of the source", "dimensionless"),
            ]

        return dict(library=l, observed=o, emission_lines=e, other=x)

    def predict(self, theta, observations=None, sps=None, **extras):
        """Given a ``theta`` vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        Parameters
        ----------
        theta : ndarray of shape ``(ndim,)``
            Vector of free model parameter values.

        observations : A list of `Observation` instances (e.g. instance of Photometry())
            The data to predict

        sps :
            An `sps` object to be used in the model generation.  It must have
            the :py:func:`get_galaxy_spectrum` method defined.

        Returns
        -------
        predictions: (list of ndarrays)
            List of predictions for the given list of observations.

            If the observation kind is "spectrum" then this is the model spectrum for these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            including multiplication by the calibration vector.  Units of
            maggies

            If the observation kind is "photometry" then this is the model
            photometry for these parameters, for the filters specified in
            ``obs['filters']``.  Units of maggies.

        extras :
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            stellar mass formed.
        """
        self.predict_init(theta, sps)

        # generate predictions for likelihood
        # this assumes all spectral datasets (if present) occur first
        # because they can change the line strengths during marginalization.
        predictions = [self.predict_obs(obs) for obs in observations]

        return predictions, self._mfrac

    def predict_init(self, theta, sps):
        """Generate the physical model on the model wavelength grid, and cache
        many quantities used in common for all kinds of predictions.

        Parameters
        ----------
        theta : ndarray of shape ``(ndim,)``
            Vector of free model parameter values.

        sps :
            An `sps` object to be used in the model generation.  It must have
            the :py:func:`get_galaxy_spectrum` method defined.
        """
        # generate and cache intrinsic model spectrum and info
        self.set_parameters(theta)
        self._wave, self._spec, self._mfrac = sps.get_galaxy_spectrum(**self.params)
        self._zred = self.params.get('zred', 0)
        self._eline_wave, self._eline_lum = sps.get_galaxy_elines()
        self._library_resolution = getattr(sps, "spectral_resolution", 0.0) # restframe

        # Flux normalize
        self._norm_spec = self._spec * self.flux_norm()

        # cache eline observed wavelengths
        eline_z = self.params.get("eline_delta_zred", 0.0)
        self._ewave_obs = (1 + eline_z + self._zred) * self._eline_wave

        # cache eline mle info
        self._ln_eline_penalty = 0
        self._eline_lum_mle = self._eline_lum.copy()
        self._eline_lum_covar = np.diag((self.params.get('eline_prior_width', 0.0) *
                                         self._eline_lum)**2)

        # physical velocity smoothing of the whole UV/NIR spectrum
        self._smooth_spec = self.losvd_smoothing(self._wave, self._norm_spec)

        # Ly-alpha absorption
        self._smooth_spec = self.add_dla(self._wave, self._smooth_spec)
        self._smooth_spec = self.add_damping_wing(self._wave, self._smooth_spec)

    def predict_obs(self, obs):
        if obs.kind == "spectrum":
            prediction = self.predict_spec(obs)
        elif obs.kind == "lines":
            prediction = self.predict_lines(obs)
        elif obs.kind == "photometry":
            prediction = self.predict_phot(obs.filterset)
        elif obs.kind == "intrinsic":
            prediction = self.predict_intrinsic(obs)
        else:
            prediction = None
        return prediction

    def predict_spec(self, obs):
        """Generate a prediction for the observed spectrum.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct

          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies
          + ``_eline_wave`` and ``_eline_lum`` - emission line parameters from the SPS model

        It generates the following attributes

          + ``_outwave`` - Wavelength grid (instrument frame)
          + ``_speccal`` - Calibration vector
          + ``_sed`` - Intrinsic spectrum (before cilbration vector applied but including emission lines)

        And the following attributes are generated if nebular lines are added

          + ``_fix_eline_spec`` - emission line spectrum for fixed lines, intrinsic units
          + ``_fix_eline_spec`` - emission line spectrum for fitted lines, with
            spectroscopic calibration factor included.

        Numerous quantities related to the emission lines are also cached (see
        ``cache_eline_parameters()`` and ``fit_mle_elines()`` for details.)

        Parameters
        ----------
        obs : Instance of :py:class:`observation.Spectrum`
            Must contain the output wavelength array, the observed fluxes and
            uncertainties thereon.

        sigma_spec : (optional)
            The covariance matrix for the spectral noise. It is only used for
            emission line marginalization.

        Returns
        -------
        spec : ndarray of shape ``(nwave,)``
            The prediction for the observed frame spectral flux these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            including multiplication by the calibration vector. in units of
            maggies.
        """
        # redshift model wavelength
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)

        # get output wavelength vector
        # TODO: remove this and require all Spectrum instances to have a wavelength array
        self._outwave = obs.wavelength
        if self._outwave is None:
            self._outwave = obs_wave

        # Set up for emission lines
        self.cache_eline_parameters(obs)

        # --- smooth and put on output wavelength grid ---
        # Instrumental smoothing (accounting for library resolution)
        # Put onto the spec.wavelength grid.

        # HACK to change the spectral resolution on the fly
        if hasattr(obs, "resolution_jitter_parameter"):
            parn = getattr(obs, "resolution_jitter_parameter")
            res_jitter = self.params.get(parn)
            obs.padded_resolution = np.interp(obs.padded_wavelength,
                                              obs.wavelength,
                                              obs.resolution * res_jitter)

        inst_spec = obs.instrumental_smoothing(obs_wave, self._smooth_spec,
                                               libres=self._library_resolution)

        # --- add fixed lines if necessary ---
        emask = self._fix_eline_pixelmask
        if emask.any():
            inds = self._fix_eline & self._valid_eline
            espec = self.predict_eline_spec(line_indices=inds,
                                            wave=self._outwave[emask])
            self._fix_eline_spec = espec
            inst_spec[emask] += self._fix_eline_spec.sum(axis=1)

        # --- (de-) apply calibration ---
        extra_mask = self._fit_eline_pixelmask
        if not extra_mask.any():
            extra_mask = True  # all pixels are ok
        response = obs.compute_response(spec=inst_spec,
                                        extra_mask=extra_mask,
                                        **self.params)
        inst_spec = inst_spec * response

        # --- fit and add lines if necessary ---
        emask = self._fit_eline_pixelmask
        if emask.any():
            # We need the spectroscopic covariance matrix to do emission line
            # optimization and marginalization
            spec_unc = None
            # FIXME: do this only if the noise model is non-trivial, and make sure masking is consistent
            #vectors = obs.noise.populate_vectors(obs)
            #spec_unc = obs.noise.construct_covariance(**vectors)
            self._fit_eline_spec = self.fit_mle_elines(obs, inst_spec, spec_unc)
            inst_spec[emask] += self._fit_eline_spec.sum(axis=1)

        # --- cache intrinsic spectrum for this observation ---
        self._sed = inst_spec / response
        self._speccal = response

        return inst_spec

    def predict_lines(self, obs, **extras):
        """Generate a prediction for the observed nebular line fluxes.  This method assumes
        that the model parameters have been set, that any adjustments to the
        emission line fluxes based on ML fitting have been applied, and that the
        following attributes are present and correct
          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_eline_wave`` and ``_eline_lum`` - emission line parameters from the SPS model
        It generates the following attributes
          + ``_outwave`` - Wavelength grid (observed frame)
          + ``_speccal`` - Calibration vector
          + ``line_norm`` - the conversion from FSPS line luminosities to the
                            observed line luminosities, including scaling fudge_factor
          + ``_predicted_line_inds`` - the indices of the line that are predicted

        Numerous quantities related to the emission lines are also cached (see
        ``cache_eline_parameters()`` and ``fit_mle_elines()`` for details) including
        ``_predicted_line_inds`` which is the indices of the line that are predicted.
        ``cache_eline_parameters()`` and ``fit_elines()`` for details).

        Parameters
        ----------
        obs : Instance of :py:class:``observation.Lines``
            Must have the attributes:
            + ``"wavelength"`` - the observed frame wavelength of the lines.
            + ``"line_ind"`` - a set of indices identifying the observed lines in
            the fsps line array

        Returns
        -------
        elum : ndarray of shape ``(nwave,)``
            The prediction for the observed frame nebular emission line flux
            these parameters, at the wavelengths specified by
            ``obs['wavelength']``, in units of erg/s/cm^2.
        """
        obs_wave = self.observed_wave(self._eline_wave, do_wavecal=False)
        self._outwave = obs.get('wavelength', obs_wave)
        assert len(self._outwave) <= len(self.emline_info)

        # --- cache eline parameters ---
        self.cache_eline_parameters(obs)

        # find the indices of the observed emission lines
        #dw = np.abs(self._ewave_obs[:, None] - self._outwave[None, :])
        #self._predicted_line_inds = np.argmin(dw, axis=0)
        self._predicted_line_inds = obs["line_inds"]
        self._speccal = 1.0

        self.line_norm = self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs)
        self.line_norm *= self.params.get("linespec_scaling", 1.0)
        elums = self._eline_lum[self._predicted_line_inds] * self.line_norm

        return elums

    def predict_phot(self, filterset):
        """Generate a prediction for the observed photometry.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct:
          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies.
          + ``_ewave_obs`` and ``_eline_lum`` - emission line parameters from
            the SPS model

        Parameters
        ----------
        filters : Instance of :py:class:`sedpy.observate.FilterSet` or list of
            :py:class:`sedpy.observate.Filter` objects. If there is no
            photometry, ``None`` should be supplied.

        Returns
        -------
        phot : ndarray of shape ``(len(filters),)``
            Observed frame photometry of the model SED through the given filters,
            in units of maggies. If ``filters`` is None, this returns 0.0
        """
        if filterset is None:
            return 0.0

        # generate photometry w/o emission lines
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        flambda = self._smooth_spec * lightspeed / obs_wave**2 * (3631*jansky_cgs)
        phot = np.atleast_1d(getSED(obs_wave, flambda, filterset, linear_flux=True))

        # generate emission-line photometry
        if (self._want_lines & self._need_lines):
            phot += self.nebline_photometry(filterset)

        return phot

    def flux_norm(self):
        """Compute the scaling required to go from Lsun/Hz/Msun to maggies.
        Note this includes the (1+z) factor required for flux densities.

        Returns
        -------
        norm : (float)
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

    def init_eline_info(self, eline_file='emlines_info.dat'):

        # get the emission line info
        try:
            SPS_HOME = os.getenv('SPS_HOME')
            info = np.genfromtxt(os.path.join(SPS_HOME, 'data', eline_file),
                                 dtype=[('wave', 'f8'), ('name', '<U20')],
                                 delimiter=',')
            self.emline_info = info
            self._use_eline = np.ones(len(info), dtype=bool)
        except(OSError, KeyError, ValueError) as e:
            print("Could not read and cache emission line info from $SPS_HOME/data/emlines_info.dat")
            self.emline_info = e

    @property
    def _need_lines(self):
        return (not (bool(np.any(self.params.get("nebemlineinspec", True)))))

    @property
    def _want_lines(self):
        return bool(np.all(self.params.get('add_neb_emission', False)))

    def nebline_photometry(self, filterset, elams=None, elums=None):
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
        flux = np.zeros(len(filterset))
        try:
            # TODO: Since in this case filters are on a grid, there should be a
            # faster way to look up the transmission than the later loop
            flist = filterset.filters
        except(AttributeError):
            flist = filterset
        for i, filt in enumerate(flist):
            # calculate transmission at line wavelengths
            trans = np.interp(elams, filt.wavelength, filt.transmission,
                              left=0., right=0.)
            # include all lines where transmission is non-zero
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*elams[idx]*elums[idx]).sum() / filt.ab_zero_counts

        return flux

    def cache_eline_parameters(self, obs, nsigma=5, forcelines=False):
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

        N.B. This must be run separately for each `Observation` instance at each
        likelihood call!!!

        :param obs: observation.Spectrum() subclass
            If given, provides the instrumental resolution for broadening the
            emission lines.

        :param nsigma: (float, optional, default: 5.)
            Number of sigma from a line center to use for defining which lines
            to fit and useful spectral elements for the fitting.  float.
        """
        # exit gracefully if not adding lines.  We also exit if only fitting
        # photometry, for performance reasons
        hasspec = obs.get('spectrum', None) is not None
        #hasspec = True
        if not (self._want_lines & self._need_lines & hasspec):
            self._fit_eline_pixelmask = np.array([], dtype=bool)
            self._fix_eline_pixelmask = np.array([], dtype=bool)
            return

        # linewidths
        nline = self._ewave_obs.shape[0]
        # physical * library linewidths
        lib_sigma_kms = np.abs(np.interp(self._eline_wave, self._wave, self._library_resolution))
        losvd = self.params.get("sigma_smooth", 300)
        self._eline_sigma_kms = np.atleast_1d(self.params.get('eline_sigma', losvd))
        # add the library resolution in quadrature; this mimics the smoothing done to
        # the stellar continuum
        self._eline_sigma_kms = np.hypot(self._eline_sigma_kms, lib_sigma_kms)
        # what is this wierd construction for?
        #self._eline_sigma_kms = (self._eline_sigma_kms[None] * np.ones(nline)).squeeze()
        #self._eline_sigma_lambda = eline_sigma_kms * self._ewave_obs / ckms
        # instrumental linewidths
        if obs.resolution is not None:
            # add the difference betwen the library and instrumental resolution in quadrature
            sigma_inst = np.interp(self._ewave_obs, obs.wavelength, obs.resolution)
            # TODO: this allows R_inst > R_lib if physical velocity dispersion is high - use if we change how stars are treated.
            #self._eline_sigma_kms = np.sqrt(self._eline_sigma_kms**2 + sigma_inst**2 - lib_sigma_kms**2)
            delta_sigma = np.sqrt(sigma_inst**2 - lib_sigma_kms**2)
            # TODO: raise a warning here?
            delta_sigma[~np.isfinite(delta_sigma)] = 0.0
            self._eline_sigma_kms = np.hypot(self._eline_sigma_kms, delta_sigma)

        # --- get valid lines ---
        # fixed and fit lines specified by user, but remove any lines which do
        # not have an observed pixel within 5sigma of their center
        # This part has to go in every call
        linewidth = nsigma * self._ewave_obs / ckms * self._eline_sigma_kms
        pixel_mask = (np.abs(self._outwave - self._ewave_obs[:, None]) < linewidth[:, None])
        omask = obs.get("mask", None)
        if omask is not None:
            pixel_mask = pixel_mask & omask
        self._valid_eline = pixel_mask.any(axis=1) & self._use_eline

        # --- wavelengths corresponding to valid lines ---
        # within N sigma of the central wavelength
        self._fit_eline_pixelmask = pixel_mask[self._valid_eline & self._fit_eline, :].any(axis=0)
        self._fix_eline_pixelmask = pixel_mask[self._valid_eline & self._fix_eline, :].any(axis=0)
        # --- lines to fit ---
        self._elines_to_fit = self._fit_eline & self._valid_eline

    def parse_elines(self):
        """Create mask arrays to identify the lines that should be fit and the
        lines that should be fixed based on the content of `params["elines_to_fix"]`
        and `params["elines_to_fit"]`

        This can probably be cached just once at instantiation unless you want
        to change between separate likelihood calls.
        """

        all_lines = self.emline_info['name']

        if self.params.get('marginalize_elines', False):
            # if marginalizing, default to fitting all lines
            # unless some are explicitly fixed
            lnames_to_fit = self.params.get('elines_to_fit', all_lines)
            lnames_to_fix = self.params.get('elines_to_fix', np.array([]))
            assert np.all(np.isin(lnames_to_fit, all_lines)), f"Some lines to fit ({lnames_to_fit})are not in the cloudy grid; see $SPS_HOME/data/emlines_info.dat for accepted names"
            assert np.all(np.isin(lnames_to_fix, all_lines)), f"Some fixed lines ({lnames_to_fix}) are not in the cloudy grid; see $SPS_HOME/data/emlines_info.dat for accepted names"
            self._fit_eline = np.isin(all_lines, lnames_to_fit) & ~np.isin(all_lines, lnames_to_fix)
        else:
            self._fit_eline = np.zeros(len(all_lines), dtype=bool)

        self._fix_eline = ~self._fit_eline

        if "elines_to_ignore" in self.params:
            assert np.all(np.isin(self.params["elines_to_ignore"], self.emline_info["name"])), f"Some ignored lines lines ({self.params['elines_to_ignore']}) are not in the cloudy grid; see $SPS_HOME/data/emlines_info.dat for accepted names"

            self._use_eline = ~np.isin(self.emline_info["name"],
                                       self.params["elines_to_ignore"])

    def fit_mle_elines(self, obs, calibrated_spec, sigma_spec=None):
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
            include fixed lines but not lines to be fit

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
        assert self._need_lines
        assert self._want_lines

        # generate Gaussians on appropriate wavelength grid
        idx = self._elines_to_fit
        emask = self._fit_eline_pixelmask
        nebwave = self._outwave[emask]
        eline_gaussians = self.get_eline_gaussians(lineidx=idx, wave=nebwave)

        # generate residuals
        delta = obs['spectrum'][emask] - calibrated_spec[emask]

        # generate line amplitudes in observed flux units
        units_factor = self.flux_norm() / (1 + self._zred)
        # FIXME: use obs.response instead of _speccal, remove all references to speccal
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

        # Calculate ML emission line amplitudes and covariance matrix
        # FIXME: nebopt: do this with a solve
        sigma_alpha_hat = np.linalg.pinv(np.dot(eline_gaussians.T, np.dot(sigma_inv, eline_gaussians)))
        alpha_hat = np.dot(sigma_alpha_hat, np.dot(eline_gaussians.T, np.dot(sigma_inv, delta)))

        # Generate likelihood penalty term (and MAP amplitudes)

        # grab current covar matrix for these lines
        sigma_alpha_breve = self._eline_lum_covar[np.ix_(idx, idx)]

        if np.any(sigma_alpha_breve > 0):
            # Incorporate gaussian "priors" on the amplitudes
            # these can come from cloudy model & uncertainty, or fit from previous dataset

            # first account for calibration vector
            sigma_alpha_breve = sigma_alpha_breve * linecal[:, None] * linecal[None, :]

            # combine covar matrices
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
            sigma_alpha_bar = sigma_alpha_hat
            K = ln_mvn(alpha_hat, mean=alpha_hat, cov=sigma_alpha_hat)

        # Cache the ln-penalty, accumulating (in case there are multiple spectra)
        self._ln_eline_penalty += K

        # Store fitted emission line luminosities in physical units, including prior
        self._eline_lum[idx] = alpha_bar / linecal
        # store new Gaussian uncertainties in physical units
        self._eline_lum_covar[np.ix_(idx, idx)] = sigma_alpha_bar / linecal[:, None] / linecal[None, :]

        # return the maximum-likelihood line spectrum for this observation in observed units
        self._eline_lum_mle[idx] = alpha_hat / linecal
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

    def losvd_smoothing(self, wave, spec):
        """Smooth the spectrum in velocity space.
        See :py:func:`prospect.utils.smoothing.smoothspec` for details.
        """
        sigma = self.params.get("sigma_smooth", 300)
        sel = (wave > 0.912e3) & (wave < 2.5e4)
        # TODO: make a fast version of this that is also accurate
        sm = smoothspec(wave, spec, sigma, outwave=wave[sel],
                        smoothtype="vel", fftsmooth=True)
        outspec = spec.copy()
        outspec[sel] = sm

        return outspec

    def add_dla(self, wave_rest, spec):
        logN = self.params.get("dla_logNh", None)
        if logN is None:
            return spec
        # Shift spectrum to the restframe of the DLA
        dla_z = self.params.get("dla_redshift", self._zred)
        if dla_z > self._zred:
            return spec
        wave_rest = wave_rest * (1 + dla_z) / (1 + self._zred)
        tau = voigt_profile(wave_rest, 10**logN)
        spec *= np.exp(-tau)
        return spec

    def add_damping_wing(self, wave_rest, spec):
        zmin = 5.0
        if (self._zred > zmin) & np.any(self.params.get("igm_damping", False)):
            x_HI = self.params.get("igm_factor", 1.0)
            tau = tau_damping(wave_rest, self._zred, x_HI, zmin=zmin, cosmo=cosmo)
            spec *= np.exp(-tau)
        return spec

    def observed_wave(self, wave, do_wavecal=False):
        """Convert the restframe wavelength grid to the observed frame wavelength
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

    def absolute_rest_maggies(self, filterset):
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
        abs_rest_maggies = np.atleast_1d(getSED(self._wave, flambda, filterset, linear_flux=True))

        # add emission lines
        if (self._want_lines & self._need_lines):
            eline_z = self.params.get("eline_delta_zred", 0.0)
            elams = (1 + eline_z) * self._eline_wave
            elums = self._eline_lum * self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs) * (ld / 10)**2
            emaggies = self.nebline_photometry(filterset, elams=elams, elums=elums)
            abs_rest_maggies += emaggies

        return abs_rest_maggies


class AGNSpecModel(SpecModel):

    # TODO: simplify this to use SpecModel methods
    # main difference is nsigma based on agn_eline_sigma

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.init_aline_info()

    def _available_parameters(self):
        pars = [("agn_elum",
                 "This float scales the predicted AGN nebular emission line"
                 "luminosities, in units of Lsun(Hbeta)/Mformed"),
                ("agn_eline_sigma",
                 "This float gives the velocity dispersion of the AGN emission"
                 "lines in km/s")
                ]

        return pars

    def init_aline_info(self):
        """AGN line spectrum. Based on data as reported in Richardson et al.
        2014 (Table 3, the 'a42' dataset) and normalized to Hbeta.

        index=59 is Hbeta
        """
        ainds = np.array([38, 40, 41, 43, 45, 50, 51, 52, 59,
                          61, 62, 64, 68, 69, 70, 72, 73, 74,
                          75, 76, 77, 78, 80])
        afluxes = np.array([2.96, 0.06, 0.1 , 1.  , 0.2 , 0.25, 0.48, 0.13, 1.,
                            2.87, 8.53, 0.07, 0.02, 0.1 , 0.33, 0.09, 0.79, 2.86,
                            2.13, 0.03, 0.77, 0.65, 0.19])
        self._aline_lum = np.zeros(len(self.emline_info))
        assert np.abs(self.emline_info["wave"][59] - 4863) < 2
        self._aline_lum[ainds] = afluxes

    def predict_spec(self, obs):
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
        ``cache_eline_parameters()`` and ``fit_mle_elines()`` for details.)

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the observed fluxes and
            uncertainties thereon.

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
        nsigma = 5 * np.max(self.params.get("agn_eline_sigma", 100.0) / self.params.get("eline_sigma", 100))
        self.cache_eline_parameters(obs, nsigma=nsigma)

        # --- smooth and put on output wavelength grid ---
        smooth_spec = self.losvd_smoothing(obs_wave, self._norm_spec)
        smooth_spec = obs.instrumental_smoothing(obs_wave, smooth_spec,
                                                 libres=self._library_resolution)

        # --- add fixed lines ---
        assert self._need_lines, "must add agn and nebular lines within prospector"
        assert not np.any(self.params.get("marginalize_elines", False)), "Cannot fit lines when AGN lines included"

        emask = self._fix_eline_pixelmask
        if emask.any():
            # Add SF lines
            inds = self._fix_eline & self._valid_eline
            espec = self.predict_eline_spec(line_indices=inds,
                                            wave=self._outwave[emask])
            self._fix_eline_spec = espec
            smooth_spec[emask] += self._fix_eline_spec.sum(axis=1)

            # Add agn lines
            aspec = self.predict_aline_spec(line_indices=inds,
                                            wave=self._outwave[emask])
            self._agn_eline_spec = aspec
            smooth_spec[emask] += self._agn_eline_spec

        # --- calibration ---
        response = obs.compute_response(spec=smooth_spec, **self.params)
        inst_spec = smooth_spec * response

        # --- cache intrinsic spectrum ---
        self._sed = inst_spec / response
        self._speccal = response

        return inst_spec

    def predict_lines(self, obs, **extras):
        """Generate a prediction for the observed nebular line fluxes, including
        AGN.

        :param obs:
            A ``data.observation.Lines()`` instance, with the attributes
            + ``"wavelength"`` - the observed frame wavelength of the lines.
            + ``"line_ind"`` - a set of indices identifying the observed lines in
            the fsps line array

        :returns elum:
            The prediction for the observed frame nebular + AGN emission line
            flux these parameters, at the wavelengths specified by
            ``obs['wavelength']``, ndarray of shape ``(nwave,)`` in units of
            erg/s/cm^2.
        """
        sflums = super().predict_lines(obs, **extras)
        anorm = self.params.get('agn_elum', 1.0) * self.line_norm
        alums = self._aline_lum[self._predicted_line_inds] * anorm

        elums = sflums + alums

        return elums

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
            # Add agn lines to photometry
            # this could use _use_line
            anorm = self.params.get('agn_elum', 1.0) * self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs)
            alums = self._aline_lum * anorm
            alams = self._ewave_obs
            phot += self.nebline_photometry(filters, alams, alums)

        return phot

    def predict_aline_spec(self, line_indices, wave):
        # HACK to change the AGN line widths.
        orig = self._eline_sigma_kms
        nline = self._ewave_obs.shape[0]
        self._eline_sigma_kms = np.atleast_1d(self.params.get('agn_eline_sigma', 100.0))
        self._eline_sigma_kms = (self._eline_sigma_kms[None] * np.ones(nline)).squeeze()
        #self._eline_sigma_kms *= np.ones(self._ewave_obs.shape[0])
        gaussians = self.get_eline_gaussians(lineidx=line_indices, wave=wave)
        self._eline_sigma_kms = orig

        anorm = self.params.get('agn_elum', 1.0) * self.flux_norm() / (1 + self._zred)
        alums = self._aline_lum[line_indices] * anorm
        aline_spec = (alums * gaussians).sum(axis=1)
        return aline_spec


class AGNPolySpecModel(SpecModel):
    """AGN acceration disk continuum model.
    For details, see Wang et al. 2024: https://ui.adsabs.harvard.edu/abs/2024arXiv240302304W/abstract
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.init_eline_info()

    def predict_init(self, theta, sps=None, **extras):

        self.set_parameters(theta)
        self._library_resolution = getattr(sps, "spectral_resolution", 0.0) # restframe

        self._wave, self._galspec, self._agnspec, self._agnspec_torus, self._mfrac, other = sps.get_galaxy_spectrum(**self.params)
        self._zred = self.params.get('zred', 0)
        self._eline_wave, self._eline_lum = sps.get_galaxy_elines()

        self._norm_galspec = self._galspec * self.flux_norm()
        self._norm_agnspec = self._agnspec * self.flux_norm()
        self._norm_agnspec_torus = self._agnspec_torus * self.flux_norm()

        cache_intr_spec = self.params.get('cache_intrinsic_spec', False)
        if cache_intr_spec:
            self._galspec_nodust = other['gal_tot_nodust']
            self._agnspec_nodust = other['agn_nodust']

            self._galspec_nodust *= self.flux_norm()
            self._agnspec_nodust *= self.flux_norm()

        self._norm_spec = self._norm_galspec + self._norm_agnspec + self._norm_agnspec_torus

        # cache eline observed wavelengths
        eline_z = self.params.get("eline_delta_zred", 0.0)
        self._ewave_obs = (1 + eline_z + self._zred) * self._eline_wave

        # cache eline mle info
        self._ln_eline_penalty = 0
        self._eline_lum_mle = self._eline_lum.copy()
        self._eline_lum_covar = np.diag((self.params.get('eline_prior_width', 0.0) *
                                         self._eline_lum)**2)

        # physical velocity smoothing of the whole UV/NIR spectrum
        self._smooth_spec = self.losvd_smoothing(self._wave, self._norm_spec)

        # Ly-alpha absorption
        self._smooth_spec = self.add_dla(self._wave, self._smooth_spec)
        self._smooth_spec = self.add_damping_wing(self._wave, self._smooth_spec)


class HyperSpecModel(ProspectorHyperParams, SpecModel):
    pass


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


# TODO: Move the below to a separate IGM module


def H(a, x):
    """Voigt Profile Approximation from T. Tepper-Garcia (2006, 2007).
    Valid to a fractional error of ~ 1e-7 * (N_h/10^22) for Lyman-alpha (a~1e-4)"""
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def voigt_profile(wave_rest, N, bkms=40, l0=1215.6696, f=4.16e-1, gamma=6.265e8):
    """
    Calculate the optical depth Voigt profile.
    Default values of the atomic constants f, l0, and gamma are for Lyman-alpha.
    Following Krogager 2018

    Parameters
    ----------
    wave_rest : array_like, shape (N)
        Restframe wavelength grid in Angstroms at which to evaluate the optical depth.

    l0 : float
        Rest frame transition wavelength in Angstroms.

    f : float
        Oscillator strength.

    N : float
        Column density in units of cm^-2.

    bkms : float
        Velocity width of the Voigt profile in km/s.

    gamma : float
        Radiation damping constant, or Einstein constant (A_ul)

    Returns
    -------
    tau : array_like, shape (N)
        Optical depth array evaluated at the input grid wavelengths `l`.
    """
    # Units & constants
    c = 2.99792e10        # cm/s
    const = 0.0149736082  # sqrt(pi) * e**2/(c * m_e) (cgs)
    l0_cm = (l0*1.e-8)
    b = bkms * 1e5

    # Calculate Profile
    C_a = const * f * l0_cm / b
    a = l0_cm * gamma / (4.*np.pi*b)

    x = (c / b) * (1. - l0 / wave_rest)
    tau = np.float64(C_a) * N * H(a, x)

    return tau


def Voigt(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    from scipy.special import wofz
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)


def tau_damping(wave_rest, zred, x_HI, zmin=5, cosmo=cosmo, Y=0.25, l0=1215.6696):
    """Compute the optical depth redward of restframe Ly-alpha due to the IGM
    damping wing.  Fiollows Mirald-Escude 1998 and Totani 2006 in assuming a
    uniform IGM below the object redshift.

    Parameters
    ----------
    wave_rest : array_like, shape (N)
        Restframe wavelength grid in Angstroms at which to evaluate the optical depth.

    zred : float
        The object redshift.  We also assume this is the maximum redshift of the
        IGM integral

    x_HI : float
        The neutral fraction of the IGM.  Can be greater than 1 to approximate
        local overdensity.

    zmin : float
        The minimum redshift for the uniform IGM integral

    cosmo : astropy.cosmology.Cosmology() instance
        The cosmology to use for the calculation

    Y : float
        The helium fraction of the universe

    l0 : float
        Rest frame transition wavelength in Angstroms.

    Returns
    -------
    tau_damping : array_like, shape (N)
        The optical depth due to the damping wing.  Will be zero blueward of l0.

    """
    R_alpha = 2.02e-8

    wave_obs = wave_rest * (1 + zred)
    zobs = wave_obs / l0 - 1
    red = (zobs - zred) / (1+zobs) > (100 * R_alpha)

    tau_damp = np.zeros_like(wave_rest)
    xx = Ix((1 + zred) / (1 + zobs[red])) - Ix((1 + zmin) / (1 + zobs[red]))

    tau = R_alpha/np.pi * x_HI * tau_gp(cosmo, zred, Y=Y)
    tau = tau * ((1 + zobs[red]) / (1 + zred))**(3/2) * xx
    tau_damp[red] = tau
    return tau_damp


def tau_gp(cosmo, zred, Y=0.25):
    # totani 2006
    # with scaling by omega_baryon * (1-Y) * h / sqrt(Omega_m)
    #factor = 3.88e5
    #scale = ((1-Y)/0.75) * (cosmo.Ob0/0.044) * (cosmo.Om0/0.27)**(-1/2) * (cosmo.h/0.71)

    # Also Miralda-Escude substituting 1/sqrt(Omega_m) = Ho/Hz * (1+z)^(3/2))
    factor = 2.6e5
    scale = (1-Y) * cosmo.h * cosmo.Om0**(-0.5) * (cosmo.Ob0/0.03)

    # Hertz 2024 relies on Becker 01/wikipedia and is not the same (missing 1-Y ?)
    # factor = 1.8
    # scale = cosmo.h * cosmo.Om0**(-0.5) * (cosmo.Ob0/0.02)

    tau_gp = factor * scale * ((1 + zred) / 7)**(3/2)
    return tau_gp


def Ix(x):
    v = x**(9/2)/(1-x) + 9/7 * x**(3.5) + 9/5*x**(2.5) + 3*x**(1.5) + 9 * x**(0.5)
    v -= 9/2 * np.log((1+x**0.5) / (1-x**0.5))
    return v