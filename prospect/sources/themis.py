"""
Themis (Jones et al. 2017) dust emission templates.

This module provides:
- Template loading and caching
- Interpolation in qhac space
- Energy normalization (templates normalized to 1W emitted)
- Two-component model: (1-gamma)*delta_Umin + gamma*powerlaw(Umin,Umax,alpha)

Themis (The Heterogeneous dust Evolution Model for Interstellar Solids) is
an alternative dust model to Draine & Li 2007, developed within the DustPedia
framework. It uses a different dust composition with hydrogenated amorphous
carbon (HAC) grains instead of PAHs.

Key parameters:
- qhac: Mass fraction of hydrocarbon solids (HAC), analogous to qpah [0.02 - 0.40]
- umin: Minimum radiation field intensity [0.10 - 80.0]
- alpha: Power-law slope for dU/dM distribution [1.0 - 3.0]
- gamma: Fraction of dust in PDR component [0 - 1]
- umax: Fixed at 1e7 (not configurable)

Reference:
    Jones, A.P. et al. (2017), A&A, 602, A46
"""

import os
import numpy as np

__all__ = ["ThemisTemplates"]

# Speed of light in Angstrom/s
C_AA = 2.998e18


class ThemisTemplates:
    """
    Manager for Themis dust emission templates.

    This class loads pre-computed Themis templates and provides
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
        default location (prospect/sources/dust_data/themis/templates.npz)

    Attributes
    ----------
    wavelength : ndarray
        Template wavelength grid in Angstroms
    qhac_values : ndarray
        Available qhac values [0.02 - 0.40]
    umin_values : ndarray
        Available Umin values [0.10 - 80.0]
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
                'dust_data', 'themis', 'templates.npz'
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
                f"Themis template file not found: {filepath}\n"
                "Run scripts/build_themis_templates.py to generate templates."
            )

        data = np.load(filepath)
        self.wavelength = data['wavelength']  # Angstroms
        self.qhac_values = data['qhac_values']
        self.umin_values = data['umin_values']
        self.alpha_values = data['alpha_values']
        self._templates_minmin = data['templates_minmin']  # (qhac, umin, wave)
        self._templates_minmax = data['templates_minmax']  # (qhac, umin, alpha, wave)

    def get_template(self, qhac, umin, alpha, gamma, target_wave=None):
        """
        Get Themis dust emission spectrum for given parameters.

        Parameters
        ----------
        qhac : float
            HAC mass fraction [0.02 - 0.40]
        umin : float
            Minimum radiation field intensity [0.10 - 80.0]
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
        # Find bracketing qhac indices for interpolation
        qhac_idx = np.searchsorted(self.qhac_values, qhac)
        qhac_idx = np.clip(qhac_idx, 1, len(self.qhac_values) - 1)

        # Find nearest umin and alpha indices
        umin_idx = np.argmin(np.abs(self.umin_values - umin))
        alpha_idx = np.argmin(np.abs(self.alpha_values - alpha))

        # Linear interpolation in qhac
        q0 = self.qhac_values[qhac_idx - 1]
        q1 = self.qhac_values[qhac_idx]
        if q1 != q0:
            f = (qhac - q0) / (q1 - q0)
            f = np.clip(f, 0.0, 1.0)  # Clamp to avoid extrapolation issues
        else:
            f = 0.0

        spec_minmin = (1 - f) * self._templates_minmin[qhac_idx - 1, umin_idx, :] + \
                      f * self._templates_minmin[qhac_idx, umin_idx, :]
        spec_minmax = (1 - f) * self._templates_minmax[qhac_idx - 1, umin_idx, alpha_idx, :] + \
                      f * self._templates_minmax[qhac_idx, umin_idx, alpha_idx, :]

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

    def compute_umean(self, umin, alpha, gamma, umax=1e7):
        """
        Compute mean radiation field intensity <U>.

        Following similar equations to DL2007/DL2014.

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
            Nearest available alpha value
        """
        idx = np.argmin(np.abs(self.alpha_values - alpha))
        return self.alpha_values[idx]

    def get_nearest_umin(self, umin):
        """
        Get the nearest available umin value.

        Parameters
        ----------
        umin : float
            Requested umin value

        Returns
        -------
        umin_nearest : float
            Nearest available umin value
        """
        idx = np.argmin(np.abs(self.umin_values - umin))
        return self.umin_values[idx]
