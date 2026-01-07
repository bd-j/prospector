"""
Dale et al. (2014) dust emission templates.

This module provides the Dale2014 single-parameter dust emission model.
The model parameterizes the dust heating intensity distribution as:
    dM_d(U) ∝ U^(-alpha) dU

where alpha controls the shape of the IR SED:
- Low alpha (~0.5-1): More warm dust, stronger mid-IR emission
- High alpha (~3-4): More cold dust, stronger far-IR emission

Reference:
    Dale, D.A., et al. (2014), ApJ, 784, 83
"""

import os
import numpy as np

__all__ = ["Dale2014Templates"]

# Speed of light in Angstrom/s
C_AA = 2.998e18


class Dale2014Templates:
    """
    Manager for Dale et al. (2014) dust emission templates.

    The Dale2014 model is a single-parameter model where alpha controls
    the dust temperature distribution. Lower alpha means more warm dust
    (stronger mid-IR), higher alpha means more cold dust (stronger far-IR).

    Parameters
    ----------
    template_file : str, optional
        Path to the templates.npz file. If not specified, looks in the
        default location (prospect/sources/dust_data/dale2014/templates.npz)

    Attributes
    ----------
    wavelength : ndarray
        Template wavelength grid in Angstroms
    alpha_values : ndarray
        Available alpha values [0.0625 - 4.0]
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
                'dust_data', 'dale2014', 'templates.npz'
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
                f"Dale2014 template file not found: {filepath}\n"
                "Run scripts/build_dale2014_templates.py to generate templates."
            )

        data = np.load(filepath)
        self.wavelength = data['wavelength']  # Angstroms
        self.alpha_values = data['alpha_values']
        self._templates = data['templates']  # (n_alpha, n_wave)

    def get_template(self, alpha, target_wave=None):
        """
        Get Dale2014 dust emission spectrum for given alpha.

        Parameters
        ----------
        alpha : float
            Power-law slope for dM_d(U) ∝ U^(-alpha) distribution.
            Valid range: 0.0625 - 4.0
            - Lower alpha (~0.5-1): warmer dust, stronger mid-IR
            - Higher alpha (~3-4): cooler dust, stronger far-IR
        target_wave : ndarray, optional
            Target wavelength grid for interpolation (Angstroms).
            If provided, the spectrum is interpolated onto this grid.

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec : ndarray
            Spectrum normalized to 1W total emission (in L_sun/Hz units)
        """
        # Find nearest alpha index (templates are at discrete alpha values)
        alpha_idx = np.argmin(np.abs(self.alpha_values - alpha))

        # Get the spectrum for this alpha
        spec = self._templates[alpha_idx, :].copy()

        # Compute normalization (should already be ~1, but verify)
        nu = C_AA / self.wavelength  # Hz
        integral = -np.trapz(spec, nu)  # L_sun

        # Normalize to 1W (1 L_sun) emitted
        if integral > 0:
            spec_normalized = spec / integral
        else:
            spec_normalized = spec

        wave_out = self.wavelength
        if target_wave is not None:
            spec_normalized = np.interp(target_wave, self.wavelength, spec_normalized,
                                        left=0.0, right=0.0)
            wave_out = target_wave

        return wave_out, spec_normalized

    def get_nearest_alpha(self, alpha):
        """
        Get the nearest available alpha value.

        Parameters
        ----------
        alpha : float
            Requested alpha value

        Returns
        -------
        alpha_nearest : float
            Nearest alpha value in the template grid
        """
        idx = np.argmin(np.abs(self.alpha_values - alpha))
        return self.alpha_values[idx]
