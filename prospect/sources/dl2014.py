"""
DL2014 (updated Draine & Li 2007) dust emission templates.

This module provides:
- Template loading and caching
- Interpolation in qpah space
- Energy normalization (templates normalized to 1W emitted)
- Two-component model: (1-gamma)*delta_Umin + gamma*powerlaw(Umin,Umax,alpha)

Reference:
    Draine, B.T., & Li, A. (2007), ApJ, 657, 810-837
    with updates from 2014 (extended alpha parameter range)
"""

import os
import numpy as np

__all__ = ["DL2014Templates"]

# Speed of light in Angstrom/s
C_AA = 2.998e18


class DL2014Templates:
    """
    Manager for DL2014 dust emission templates.

    This class loads pre-computed DL2014 templates and provides
    interpolation to retrieve dust emission spectra for arbitrary
    parameter combinations.

    The templates are based on the two-component dust model:
        dust_emission = (1 - gamma) * model_minmin + gamma * model_minmax

    where:
        - model_minmin: dust heated by radiation field U = Umin (delta function)
        - model_minmax: dust with power-law radiation field distribution
                        dM/dU ∝ U^(-alpha) from Umin to Umax = 1e7

    Parameters
    ----------
    template_file : str, optional
        Path to the templates.npz file. If not specified, looks in the
        default location (prospect/sources/dust_data/dl2014/templates.npz)

    Attributes
    ----------
    wavelength : ndarray
        Template wavelength grid in Angstroms
    qpah_values : ndarray
        Available qpah values [0.47 - 7.32]
    umin_values : ndarray
        Available Umin values [0.10 - 50.0]
    alpha_values : ndarray
        Available alpha values [1.0 - 3.0]
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
                'dust_data', 'dl2014', 'templates.npz'
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
                f"DL2014 template file not found: {filepath}\n"
                "Run scripts/build_dl2014_templates.py to generate templates."
            )

        data = np.load(filepath)
        self.wavelength = data['wavelength']  # Angstroms
        self.qpah_values = data['qpah_values']
        self.umin_values = data['umin_values']
        self.alpha_values = data['alpha_values']
        self._templates_minmin = data['templates_minmin']  # (qpah, umin, wave)
        self._templates_minmax = data['templates_minmax']  # (qpah, umin, alpha, wave)

    def get_template(self, qpah, umin, alpha, gamma, target_wave=None):
        """
        Get DL2014 dust emission spectrum for given parameters.

        Parameters
        ----------
        qpah : float
            PAH mass fraction [0.47 - 7.32]
        umin : float
            Minimum radiation field intensity [0.10 - 50.0]
        alpha : float
            Power-law slope for dU/dM distribution [1.0 - 3.0]
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

        # Find nearest umin and alpha indices
        umin_idx = np.argmin(np.abs(self.umin_values - umin))
        alpha_idx = np.argmin(np.abs(self.alpha_values - alpha))

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
        spec_minmax = (1 - f) * self._templates_minmax[qpah_idx - 1, umin_idx, alpha_idx, :] + \
                      f * self._templates_minmax[qpah_idx, umin_idx, alpha_idx, :]

        # Two-component model
        spec = (1 - gamma) * spec_minmin + gamma * spec_minmax

        # Compute emissivity (integral in L_sun)
        # spec is in L_sun/Hz, integrate over frequency
        nu = C_AA / self.wavelength  # Hz
        emissivity = -np.trapz(spec, nu)  # L_sun (negative because nu decreases with wave)

        # Normalize to 1W emitted (as 1 L_sun emitted)
        if emissivity > 0:
            spec_normalized = spec / emissivity
        else:
            spec_normalized = spec

        wave_out = self.wavelength
        if target_wave is not None:
            spec_normalized = np.interp(target_wave, self.wavelength, spec_normalized,
                                         left=0.0, right=0.0)
            wave_out = target_wave

        return wave_out, spec_normalized, emissivity

    def compute_umean(self, umin, alpha, gamma, umax=1e7):
        """
        Compute mean radiation field intensity <U>.

        Following Draine & Li 2007 Eq. 6 and 15.

        Parameters
        ----------
        umin : float
            Minimum radiation field intensity
        alpha : float
            Power-law slope for dU/dM distribution
        gamma : float
            Fraction of dust in high-U component
        umax : float, optional
            Maximum radiation field intensity (default: 1e7)

        Returns
        -------
        umean : float
            Mean radiation field intensity
        """
        umean = (1.0 - gamma) * umin

        if np.isclose(alpha, 1.0):
            # Special case: alpha = 1
            umean += gamma * (umax - umin) / np.log(umax / umin)
        elif np.isclose(alpha, 2.0):
            # Special case: alpha = 2
            umean += gamma * np.log(umax / umin) / (1.0 / umin - 1.0 / umax)
        else:
            # General case
            oma = 1.0 - alpha
            tma = 2.0 - alpha
            umean += gamma * oma / tma * \
                     (umin ** tma - umax ** tma) / (umin ** oma - umax ** oma)

        return umean

    def compute_dust_mass(self, L_absorbed, qpah, umin, alpha, gamma):
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
        alpha : float
            Power-law slope
        gamma : float
            PDR fraction

        Returns
        -------
        dust_mass : float
            Dust mass in solar masses
        """
        _, _, emissivity = self.get_template(qpah, umin, alpha, gamma)
        # emissivity is in L_sun per unit mass of dust template
        # Need to convert using the normalization
        # For now, return in L_sun / emissivity which gives relative mass
        if emissivity > 0:
            return L_absorbed / emissivity
        else:
            return 0.0
