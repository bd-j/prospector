"""
DL2007 (Draine & Li 2007) dust emission templates from CIGALE.

This module provides:
- Template loading and caching
- Interpolation in qpah space
- Energy normalization (templates normalized to 1W emitted)
- Two-component model: (1-gamma)*delta_Umin + gamma*powerlaw(Umin,Umax)

This is a CIGALE-based implementation of DL2007 that allows for direct
comparison with FSPS's built-in Draine & Li 2007 dust emission.

Key differences from DL2014:
- DL2007 has variable umax (1e3, 1e4, 1e5, 1e6) instead of fixed 1e7
- DL2007 has alpha fixed at 2.0 (no alpha parameter)
- DL2007 has fewer qpah values (7: 0.47-4.58) and umin values (23: 0.10-25.0)

Reference:
    Draine, B.T. & Li, A. (2007), ApJ, 657, 810-837
"""

import os
import numpy as np

__all__ = ["DL2007Templates"]

# Speed of light in Angstrom/s
C_AA = 2.998e18


class DL2007Templates:
    """
    Manager for DL2007 dust emission templates from CIGALE.

    This class loads pre-computed DL2007 templates and provides
    interpolation to retrieve dust emission spectra for arbitrary
    parameter combinations.

    The templates are based on the two-component dust model:
        dust_emission = (1 - gamma) * model_minmin + gamma * model_minmax

    where:
        - model_minmin: dust heated by radiation field U = Umin (delta function)
        - model_minmax: dust with power-law radiation field distribution
                        dM/dU ∝ U^(-2) from Umin to Umax (alpha fixed at 2.0)

    Parameters
    ----------
    template_file : str, optional
        Path to the templates.npz file. If not specified, looks in the
        default location (prospect/sources/dust_data/dl2007/templates.npz)

    Attributes
    ----------
    wavelength : ndarray
        Template wavelength grid in Angstroms
    qpah_values : ndarray
        Available qpah values [0.47 - 4.58]
    umin_values : ndarray
        Available Umin values [0.10 - 25.0]
    umax_values : ndarray
        Available Umax values [1e3, 1e4, 1e5, 1e6]
    """

    _instance = None  # Singleton instance cache

    def __new__(cls, template_file=None):
        """Use singleton pattern to avoid reloading templates."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, template_file=None):
        """Initialize and load templates if not already done."""
        if self._initialized:
            return

        if template_file is None:
            template_file = os.path.join(
                os.path.dirname(__file__),
                'dust_data', 'dl2007', 'templates.npz'
            )

        self._load_templates(template_file)
        self._initialized = True

    def _load_templates(self, filepath):
        """
        Load templates from NPZ file.

        Parameters
        ----------
        filepath : str
            Path to templates.npz file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"DL2007 template file not found: {filepath}\n"
                "Run scripts/build_dl2007_templates.py to generate templates."
            )

        data = np.load(filepath)
        self.wavelength = data['wavelength']  # Angstroms
        self.qpah_values = data['qpah_values']
        self.umin_values = data['umin_values']
        self.umax_values = data['umax_values']
        self._templates_minmin = data['templates_minmin']  # (qpah, umin, wave)
        self._templates_minmax = data['templates_minmax']  # (qpah, umin, umax, wave)

    def get_nearest_umax_idx(self, umax):
        """
        Get nearest available umax index.

        Parameters
        ----------
        umax : float
            Requested umax value

        Returns
        -------
        idx : int
            Index of nearest available umax
        umax_actual : float
            Actual umax value used
        """
        idx = np.argmin(np.abs(self.umax_values - umax))
        return idx, self.umax_values[idx]

    def get_template(self, qpah, umin, umax, gamma, target_wave=None):
        """
        Get DL2007 dust emission spectrum for given parameters.

        Parameters
        ----------
        qpah : float
            PAH mass fraction [0.47 - 4.58]
        umin : float
            Minimum radiation field intensity [0.10 - 25.0]
        umax : float
            Maximum radiation field intensity [1e3, 1e4, 1e5, 1e6]
        gamma : float
            Fraction of dust mass in PDR component [0 - 1]
        target_wave : ndarray, optional
            Target wavelength grid for interpolation (Angstroms).
            If provided, the spectrum is interpolated onto this grid.

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec : ndarray
            Spectrum normalized to 1W total emission (in L_sun/Hz units)
        emissivity : float
            Emissivity in L_sun (integral of spec over frequency)
        """
        # Find bracketing qpah indices for interpolation
        qpah_idx = np.searchsorted(self.qpah_values, qpah)
        qpah_idx = np.clip(qpah_idx, 1, len(self.qpah_values) - 1)

        # Find nearest umin and umax indices
        umin_idx = np.argmin(np.abs(self.umin_values - umin))
        umax_idx, _ = self.get_nearest_umax_idx(umax)

        # Linear interpolation in qpah
        q0 = self.qpah_values[qpah_idx - 1]
        q1 = self.qpah_values[qpah_idx]
        if q1 != q0:
            f = (qpah - q0) / (q1 - q0)
            f = np.clip(f, 0.0, 1.0)  # Clamp to avoid extrapolation issues
        else:
            f = 0.0

        spec_minmin = (1 - f) * self._templates_minmin[qpah_idx - 1, umin_idx, :] + \
                      f * self._templates_minmin[qpah_idx, umin_idx, :]
        spec_minmax = (1 - f) * self._templates_minmax[qpah_idx - 1, umin_idx, umax_idx, :] + \
                      f * self._templates_minmax[qpah_idx, umin_idx, umax_idx, :]

        # Two-component model
        # spec is in W/nm (F_lambda) per kg of dust
        spec = (1 - gamma) * spec_minmin + gamma * spec_minmax

        # Convert from F_lambda (W/nm) to F_nu (W/Hz)
        # F_nu = F_lambda * lambda^2 / c
        # wavelength is in Angstroms, convert to nm for calculation
        wave_nm = self.wavelength / 10.0  # Angstrom to nm
        C_NM = 2.998e17  # Speed of light in nm/s
        spec_fnu_W = spec * (wave_nm ** 2) / C_NM  # W/Hz per kg of dust

        # Compute emissivity by integrating F_nu over frequency
        # This gives total power in W
        nu = C_AA / self.wavelength  # Hz
        emissivity = -np.trapz(spec_fnu_W, nu)  # W (negative because nu decreases)

        # Convert to L_sun/Hz and normalize so integral = 1 L_sun
        L_SUN = 3.846e26  # Solar luminosity in W
        spec_fnu_Lsun = spec_fnu_W / L_SUN  # L_sun/Hz per kg of dust

        if emissivity > 0:
            emissivity_Lsun = emissivity / L_SUN  # L_sun per kg of dust
            spec_normalized = spec_fnu_Lsun / emissivity_Lsun  # Normalized to 1 L_sun
        else:
            spec_normalized = spec_fnu_Lsun

        wave_out = self.wavelength
        if target_wave is not None:
            spec_normalized = np.interp(target_wave, self.wavelength, spec_normalized,
                                         left=0.0, right=0.0)
            wave_out = target_wave

        return wave_out, spec_normalized, emissivity

    def compute_umean(self, umin, gamma, umax):
        """
        Compute mean radiation field intensity <U>.

        Following Draine & Li 2007 Eq. 6 and 15, with alpha = 2.0 (fixed).

        Parameters
        ----------
        umin : float
            Minimum radiation field intensity
        gamma : float
            Fraction of dust in high-U component
        umax : float
            Maximum radiation field intensity

        Returns
        -------
        umean : float
            Mean radiation field intensity
        """
        # For alpha = 2.0 (fixed in DL2007)
        umean = (1.0 - gamma) * umin + \
                gamma * np.log(umax / umin) / (1.0 / umin - 1.0 / umax)
        return umean

    def compute_dust_mass(self, L_absorbed, qpah, umin, umax, gamma):
        """
        Compute dust mass from absorbed luminosity.

        Parameters
        ----------
        L_absorbed : float
            Absorbed luminosity in L_sun
        qpah : float
            PAH mass fraction
        umin : float
            Minimum radiation field intensity
        umax : float
            Maximum radiation field intensity
        gamma : float
            PDR fraction

        Returns
        -------
        dust_mass : float
            Dust mass in solar masses (relative)
        """
        _, _, emissivity = self.get_template(qpah, umin, umax, gamma)
        if emissivity > 0:
            return L_absorbed / emissivity
        else:
            return 0.0
