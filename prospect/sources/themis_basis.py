"""
ThemisSSPBasis - SPS basis with Themis (Jones et al. 2017) dust emission model.

This module provides the ThemisSSPBasis class, which generates galaxy
spectra using FSPS for stellar populations but replaces FSPS's built-in
Draine & Li 2007 dust emission with the Themis model from CIGALE.

Themis (The Heterogeneous dust Evolution Model for Interstellar Solids) is
an alternative dust model developed within the DustPedia framework. It uses
a different dust composition with hydrogenated amorphous carbon (HAC) grains
instead of PAHs.

Key parameters:
- themis_qhac: Mass fraction of HAC grains [0.02 - 0.40] (analogous to qpah)
- themis_umin: Minimum radiation field intensity [0.10 - 80.0]
- themis_alpha: Power-law slope for dU/dM distribution [1.0 - 3.0]
- themis_gamma: Fraction of dust in PDR component [0 - 1]
- umax: Fixed at 1e7

Reference:
    Jones, A.P. et al. (2017), A&A, 602, A46
"""

import numpy as np
from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust_with_absorption_tracking, add_igm
from .themis import ThemisTemplates

__all__ = ["ThemisSSPBasis"]


class ThemisSSPBasis(FastStepBasis):
    """
    SSP basis that uses Themis dust emission templates instead
    of FSPS's built-in dust emission.

    This class overrides get_galaxy_spectrum to:
    1. Generate stellar spectrum with add_dust_emission=False
    2. Apply dust attenuation and track absorbed luminosity
    3. Add Themis dust emission scaled to absorbed luminosity

    The Themis dust emission parameters should be provided via the
    model parameters (see TemplateLibrary["themis_dust_emission"]):
        - themis_qhac: HAC mass fraction [0.02 - 0.40]
        - themis_umin: Minimum radiation field [0.10 - 80.0]
        - themis_alpha: Power-law slope [1.0 - 3.0]
        - themis_gamma: PDR fraction [0 - 1]

    Parameters
    ----------
    zcontinuous : int, optional
        The zcontinuous parameter for FSPS (default: 1)
    reserved_params : list, optional
        Additional parameters to reserve from being passed to FSPS
    **kwargs :
        Additional keyword arguments passed to FastStepBasis

    Example
    -------
    >>> from prospect.sources import ThemisSSPBasis
    >>> from prospect.models.templates import TemplateLibrary
    >>>
    >>> # Build model with Themis dust emission
    >>> model_params = TemplateLibrary["parametric_sfh"]
    >>> model_params.update(TemplateLibrary["themis_dust_emission"])
    >>>
    >>> # Use ThemisSSPBasis instead of CSPSpecBasis
    >>> sps = ThemisSSPBasis(zcontinuous=1)
    """

    def __init__(self, zcontinuous=1, reserved_params=None, **kwargs):
        # Reserve Themis parameters from being passed to FSPS
        rp = ['themis_qhac', 'themis_umin', 'themis_alpha', 'themis_gamma']
        if reserved_params is not None:
            rp = rp + list(reserved_params)

        super().__init__(zcontinuous=zcontinuous,
                         reserved_params=rp,
                         **kwargs)

        # Force FSPS to NOT add dust emission - we'll add it ourselves
        self.ssp.params['add_dust_emission'] = False

        # Load Themis templates (singleton, cached)
        self._themis = ThemisTemplates()

    def get_galaxy_spectrum(self, **params):
        """
        Generate galaxy spectrum with Themis dust emission.

        This method:
        1. Generates a stellar spectrum from FSPS (without dust emission)
        2. Applies dust attenuation, tracking the absorbed luminosity
        3. Adds Themis dust emission scaled by the absorbed luminosity

        Parameters
        ----------
        **params : dict
            Model parameters including stellar population parameters,
            dust attenuation parameters (dust_type, dust2, etc.), and
            Themis parameters (themis_qhac, themis_umin, themis_alpha,
            themis_gamma)

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec : ndarray
            Total spectrum (stellar + dust) in L_sun/Hz per Msun formed
        mfrac : float
            Surviving mass fraction
        """
        self.update(**params)

        # Validate agebins spacing
        if np.min(np.diff(10**self.params['agebins'])) < 1e6:
            raise ValueError("Agebins must have minimum spacing of 1 Myr")

        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])

        # Get stellar spectrum WITHOUT dust emission
        self.ssp.params["sfh"] = 3
        self.ssp.params["add_dust_emission"] = False
        self.ssp.set_tabular_sfh(time, sfr)

        wave, spec_with_internal_dust = self.ssp.get_spectrum(tage=tmax, peraa=False)

        # Get young/old components (no dust applied internally)
        young, old = self.ssp._csp_young_old
        specs = [young, old]

        # Get emission lines
        ewave = self.ssp.emline_wavelengths
        eline_lum = self.ssp.emline_luminosity.copy()
        if eline_lum.ndim > 1:
            eline_lum = eline_lum[0]
        # Assume all lines come from young component
        elines = [eline_lum, np.zeros_like(eline_lum)]

        # Apply dust attenuation and track absorbed energy
        # Use FSPS defaults for dust parameters when not specified in model
        dust_type = int(self.params.get('dust_type', self.ssp.params['dust_type']))
        dust_index = float(self.params.get('dust_index', self.ssp.params['dust_index']))
        dust2 = float(self.params.get('dust2', self.ssp.params['dust2']))
        dust1_index = float(self.params.get('dust1_index', self.ssp.params['dust1_index']))
        dust1 = float(self.params.get('dust1', self.ssp.params['dust1']))
        frac_nodust = float(self.params.get('frac_nodust', 0.0))
        frac_obrun = float(self.params.get('frac_obrun', 0.0))

        attenuated_spec, attenuated_lines, L_absorbed = add_dust_with_absorption_tracking(
            wave, specs, ewave, elines,
            dust_type=dust_type, dust_index=dust_index, dust2=dust2,
            dust1_index=dust1_index, dust1=dust1,
            frac_nodust=frac_nodust, frac_obrun=frac_obrun
        )

        # Apply IGM absorption if requested
        attenuated_spec = add_igm(wave, attenuated_spec, **self.params)

        # Get Themis parameters (with reasonable defaults)
        qhac = float(self.params.get('themis_qhac', 0.17))
        umin = float(self.params.get('themis_umin', 1.0))
        alpha = float(self.params.get('themis_alpha', 2.0))
        gamma = float(self.params.get('themis_gamma', 0.1))

        # Get Themis dust emission template (normalized to 1 L_sun emitted)
        _, dust_spec_norm, emissivity = self._themis.get_template(
            qhac, umin, alpha, gamma, target_wave=wave
        )

        # Scale by absorbed luminosity
        dust_emission = L_absorbed * dust_spec_norm

        # Combine stellar + dust emission
        total_spec = attenuated_spec + dust_emission

        # Store useful quantities for diagnostics
        self._L_absorbed = L_absorbed / mtot  # per mass formed
        self._L_dust = L_absorbed / mtot  # dust luminosity = absorbed luminosity
        self._qhac = qhac
        self._umin = self._themis.get_nearest_umin(umin)  # actual umin used
        self._alpha = self._themis.get_nearest_alpha(alpha)  # actual alpha used
        self._gamma = gamma
        self._umean = self._themis.compute_umean(umin, alpha, gamma)
        self._line_specific_luminosity = attenuated_lines / mtot

        # Stellar mass fraction from FSPS
        stellar_mass_frac = self.ssp.stellar_mass / mtot

        return wave, total_spec / mtot, stellar_mass_frac

    def get_galaxy_elines(self):
        """
        Get attenuated emission line wavelengths and luminosities.

        Returns
        -------
        ewave : ndarray
            Emission line wavelengths in Angstroms
        elum : ndarray
            Specific emission line luminosities in L_sun per Msun formed
        """
        ewave = self.ssp.emline_wavelengths
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            elum = self.ssp.emline_luminosity.copy()
            if elum.ndim > 1:
                elum = elum[0]
            mass = np.sum(self.params.get('mass', 1.0))
            elum /= mass

        return ewave, elum

    @property
    def L_absorbed(self):
        """Absorbed luminosity per unit mass formed (L_sun/Msun)."""
        return getattr(self, '_L_absorbed', 0.0)

    @property
    def L_dust(self):
        """Dust emission luminosity per unit mass formed (L_sun/Msun)."""
        return getattr(self, '_L_dust', 0.0)

    @property
    def qhac(self):
        """HAC mass fraction."""
        return getattr(self, '_qhac', 0.17)

    @property
    def umin(self):
        """Minimum radiation field intensity (actual value used)."""
        return getattr(self, '_umin', 1.0)

    @property
    def alpha(self):
        """Power-law slope (actual value used)."""
        return getattr(self, '_alpha', 2.0)

    @property
    def gamma(self):
        """PDR fraction (fraction of dust in high-U component)."""
        return getattr(self, '_gamma', 0.1)

    @property
    def umean(self):
        """Mean radiation field intensity <U>."""
        return getattr(self, '_umean', 1.0)
