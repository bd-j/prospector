"""
Dale2014SSPBasis - SPS basis with Dale et al. (2014) dust emission model.

This module provides the Dale2014SSPBasis class, which generates galaxy
spectra using FSPS for stellar populations but replaces FSPS's built-in
Draine & Li 2007 dust emission with the single-parameter Dale2014 model.

The Dale2014 model parameterizes dust emission using a single parameter
alpha, where dM_d(U) ∝ U^(-alpha) dU describes the dust heating intensity
distribution. This is simpler than the DL2014/DL2007 models which have
multiple parameters.

Reference:
    Dale, D.A., et al. (2014), ApJ, 784, 83
"""

import numpy as np
from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust_with_absorption_tracking, add_igm
from .dale2014 import Dale2014Templates

__all__ = ["Dale2014SSPBasis"]


class Dale2014SSPBasis(FastStepBasis):
    """
    SSP basis that uses Dale et al. (2014) dust emission templates instead
    of FSPS's built-in dust emission.

    This class overrides get_galaxy_spectrum to:
    1. Generate stellar spectrum with add_dust_emission=False
    2. Apply dust attenuation and track absorbed luminosity
    3. Add Dale2014 dust emission scaled to absorbed luminosity

    The Dale2014 dust emission parameter should be provided via the
    model parameters (see TemplateLibrary["dale2014_dust_emission"]):
        - dale2014_alpha: Power-law slope [0.0625 - 4.0]
            Lower alpha (~0.5-1): warmer dust, stronger mid-IR
            Higher alpha (~3-4): cooler dust, stronger far-IR

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
    >>> from prospect.sources import Dale2014SSPBasis
    >>> from prospect.models.templates import TemplateLibrary
    >>>
    >>> # Build model with Dale2014 dust emission
    >>> model_params = TemplateLibrary["parametric_sfh"]
    >>> model_params.update(TemplateLibrary["dale2014_dust_emission"])
    >>>
    >>> # Use Dale2014SSPBasis instead of CSPSpecBasis
    >>> sps = Dale2014SSPBasis(zcontinuous=1)
    """

    def __init__(self, zcontinuous=1, reserved_params=None, **kwargs):
        # Reserve Dale2014 parameter from being passed to FSPS
        rp = ['dale2014_alpha']
        if reserved_params is not None:
            rp = rp + list(reserved_params)

        super().__init__(zcontinuous=zcontinuous,
                         reserved_params=rp,
                         **kwargs)

        # Force FSPS to NOT add dust emission - we'll add it ourselves
        self.ssp.params['add_dust_emission'] = False

        # Load Dale2014 templates (singleton, cached)
        self._dale2014 = Dale2014Templates()

    def get_galaxy_spectrum(self, **params):
        """
        Generate galaxy spectrum with Dale2014 dust emission.

        This method:
        1. Generates a stellar spectrum from FSPS (without dust emission)
        2. Applies dust attenuation, tracking the absorbed luminosity
        3. Adds Dale2014 dust emission scaled by the absorbed luminosity

        Parameters
        ----------
        **params : dict
            Model parameters including stellar population parameters,
            dust attenuation parameters (dust_type, dust2, etc.), and
            Dale2014 parameter (dale2014_alpha)

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

        # Get Dale2014 parameter
        alpha = float(self.params.get('dale2014_alpha', 2.0))

        # Get Dale2014 dust emission template (normalized to 1 L_sun emitted)
        _, dust_spec_norm = self._dale2014.get_template(alpha, target_wave=wave)

        # Scale by absorbed luminosity
        dust_emission = L_absorbed * dust_spec_norm

        # Combine stellar + dust emission
        total_spec = attenuated_spec + dust_emission

        # Store useful quantities for diagnostics
        self._L_absorbed = L_absorbed / mtot  # per mass formed
        self._L_dust = L_absorbed / mtot  # dust luminosity = absorbed luminosity
        self._alpha = self._dale2014.get_nearest_alpha(alpha)
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
    def alpha(self):
        """The alpha value used (nearest available in template grid)."""
        return getattr(self, '_alpha', 2.0)
