"""
DL2007CigaleSSPBasis - SPS basis with Draine & Li 2007 dust emission from CIGALE.

This module provides the DL2007CigaleSSPBasis class, which generates galaxy
spectra using FSPS for stellar populations but uses CIGALE's DL2007 templates
instead of FSPS's built-in DL2007 dust emission.

This allows for direct comparison between the two implementations and provides
the flexibility of variable umax that CIGALE supports.

Key differences from FSPS's DL2007:
- Uses CIGALE's pre-computed templates
- Variable umax (1e3, 1e4, 1e5, 1e6) instead of fixed
- alpha fixed at 2.0 (same as FSPS)

Reference:
    Draine, B.T. & Li, A. (2007), ApJ, 657, 810-837
"""

import numpy as np
from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust_with_absorption_tracking, add_igm
from .dl2007 import DL2007Templates

__all__ = ["DL2007CigaleSSPBasis"]


class DL2007CigaleSSPBasis(FastStepBasis):
    """
    SSP basis that uses CIGALE's DL2007 dust emission templates instead
    of FSPS's built-in dust emission.

    This class overrides get_galaxy_spectrum to:
    1. Generate stellar spectrum with add_dust_emission=False
    2. Apply dust attenuation and track absorbed luminosity
    3. Add DL2007 dust emission (from CIGALE) scaled to absorbed luminosity

    The DL2007 dust emission parameters should be provided via the
    model parameters (see TemplateLibrary["dl2007_cigale_dust_emission"]):
        - dl2007_cigale_qpah: PAH mass fraction [0.47 - 4.58]
        - dl2007_cigale_umin: Minimum radiation field [0.10 - 25.0]
        - dl2007_cigale_umax: Maximum radiation field [1e3, 1e4, 1e5, 1e6]
        - dl2007_cigale_gamma: PDR fraction [0 - 1]

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
    >>> from prospect.sources import DL2007CigaleSSPBasis
    >>> from prospect.models.templates import TemplateLibrary
    >>>
    >>> # Build model with DL2007 CIGALE dust emission
    >>> model_params = TemplateLibrary["parametric_sfh"]
    >>> model_params.update(TemplateLibrary["dl2007_cigale_dust_emission"])
    >>>
    >>> # Use DL2007CigaleSSPBasis instead of CSPSpecBasis
    >>> sps = DL2007CigaleSSPBasis(zcontinuous=1)
    """

    def __init__(self, zcontinuous=1, reserved_params=None, **kwargs):
        # Reserve DL2007 CIGALE parameters from being passed to FSPS
        rp = ['dl2007_cigale_qpah', 'dl2007_cigale_umin',
              'dl2007_cigale_umax', 'dl2007_cigale_gamma']
        if reserved_params is not None:
            rp = rp + list(reserved_params)

        super().__init__(zcontinuous=zcontinuous,
                         reserved_params=rp,
                         **kwargs)

        # Force FSPS to NOT add dust emission - we'll add it ourselves
        self.ssp.params['add_dust_emission'] = False

        # Load DL2007 templates (singleton, cached)
        self._dl2007 = DL2007Templates()

    def get_galaxy_spectrum(self, **params):
        """
        Generate galaxy spectrum with DL2007 CIGALE dust emission.

        This method:
        1. Generates a stellar spectrum from FSPS (without dust emission)
        2. Applies dust attenuation, tracking the absorbed luminosity
        3. Adds DL2007 dust emission scaled by the absorbed luminosity

        Parameters
        ----------
        **params : dict
            Model parameters including stellar population parameters,
            dust attenuation parameters (dust_type, dust2, etc.), and
            DL2007 CIGALE parameters (dl2007_cigale_qpah, dl2007_cigale_umin,
            dl2007_cigale_umax, dl2007_cigale_gamma)

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

        # Get DL2007 CIGALE parameters (with reasonable defaults)
        qpah = float(self.params.get('dl2007_cigale_qpah', 2.50))
        umin = float(self.params.get('dl2007_cigale_umin', 1.0))
        umax = float(self.params.get('dl2007_cigale_umax', 1e6))
        gamma = float(self.params.get('dl2007_cigale_gamma', 0.1))

        # Get DL2007 dust emission template (normalized to 1 L_sun emitted)
        _, dust_spec_norm, emissivity = self._dl2007.get_template(
            qpah, umin, umax, gamma, target_wave=wave
        )

        # Scale by absorbed luminosity
        dust_emission = L_absorbed * dust_spec_norm

        # Combine stellar + dust emission
        total_spec = attenuated_spec + dust_emission

        # Store useful quantities for diagnostics
        self._L_absorbed = L_absorbed / mtot  # per mass formed
        self._L_dust = L_absorbed / mtot  # dust luminosity = absorbed luminosity
        self._qpah = qpah
        self._umin = umin
        self._umax = self._dl2007.get_nearest_umax_idx(umax)[1]  # actual umax used
        self._gamma = gamma
        self._umean = self._dl2007.compute_umean(umin, gamma, self._umax)
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
    def qpah(self):
        """PAH mass fraction."""
        return getattr(self, '_qpah', 2.50)

    @property
    def umin(self):
        """Minimum radiation field intensity."""
        return getattr(self, '_umin', 1.0)

    @property
    def umax(self):
        """Maximum radiation field intensity (actual value used)."""
        return getattr(self, '_umax', 1e6)

    @property
    def gamma(self):
        """PDR fraction (fraction of dust in high-U component)."""
        return getattr(self, '_gamma', 0.1)

    @property
    def umean(self):
        """Mean radiation field intensity <U>."""
        return getattr(self, '_umean', 1.0)
