"""
Casey (2012) analytical dust emission model.

This module provides the Casey2012 parametric dust emission model, which
combines a modified blackbody with a mid-IR power-law component.

The model is defined by three parameters:
- temperature: Dust temperature in K
- beta: Dust emissivity index
- alpha: Mid-IR power-law slope

Unlike DL2014 and Dale2014, this model computes the SED analytically
rather than using pre-computed templates, making it very flexible.

Reference:
    Casey, C.M. (2012), MNRAS, 425, 3094
"""

import numpy as np

__all__ = ["Casey2012Model"]

# Physical constants
C_CGS = 2.998e10    # Speed of light in cm/s
C_NM = 2.998e17     # Speed of light in nm/s
C_AA = 2.998e18     # Speed of light in Angstrom/s
H_CGS = 6.626e-27   # Planck constant in erg s
K_CGS = 1.381e-16   # Boltzmann constant in erg/K
L_SUN = 3.846e26    # Solar luminosity in W


class Casey2012Model:
    """
    Casey (2012) parametric dust emission model.

    This model combines:
    1. A modified blackbody (graybody) for far-IR emission
    2. A power-law component for mid-IR emission

    The model is computed analytically, so any parameter values can be used.

    Reference:
        Casey, C.M. (2012), MNRAS, 425, 3094

    Example
    -------
    >>> model = Casey2012Model()
    >>> wave, spec = model.get_spectrum(temperature=35, beta=1.6, alpha=2.0)
    """

    def __init__(self):
        """Initialize the Casey2012 model."""
        pass

    def get_spectrum(self, temperature, beta, alpha, target_wave=None,
                     wave_min=1e3, wave_max=1e6, n_wave=1000):
        """
        Compute the Casey2012 dust emission spectrum.

        Parameters
        ----------
        temperature : float
            Dust temperature in Kelvin. Typical range: 20-60 K.
        beta : float
            Dust emissivity index. Typical range: 1.0-2.5.
            Higher beta = steeper Rayleigh-Jeans slope.
        alpha : float
            Mid-IR power-law slope. Typical range: 1.5-3.0.
            Controls the mid-IR (restframe ~10-40 μm) emission.
        target_wave : ndarray, optional
            Target wavelength grid in Angstroms. If provided, the spectrum
            is computed on this grid. Otherwise, a default grid is used.
        wave_min : float, optional
            Minimum wavelength in nm for default grid (default: 1000 nm = 1 μm)
        wave_max : float, optional
            Maximum wavelength in nm for default grid (default: 1e6 nm = 1 mm)
        n_wave : int, optional
            Number of wavelength points for default grid (default: 1000)

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec : ndarray
            Spectrum normalized to 1W (1 L_sun) total emission (in L_sun/Hz)
        """
        # Create wavelength grid in nm (internal calculation units)
        if target_wave is not None:
            wave_nm = target_wave / 10.0  # Angstrom to nm
        else:
            wave_nm = np.logspace(np.log10(wave_min), np.log10(wave_max), n_wave)

        # Model constants from Casey (2012)
        b1 = 26.68
        b2 = 6.246
        b3 = 1.905e-4
        b4 = 7.243e-5

        # Transition wavelength (nm) - where power-law meets modified blackbody
        lambda_c = 0.75e3 / ((b1 + b2 * alpha) ** -2.0 + (b3 + b4 * alpha) * temperature)

        # Reference wavelength for opacity (nm)
        lambda_0 = 200e3  # 200 μm = 200,000 nm

        # Power-law normalization factor
        # This ensures continuity between the power-law and blackbody at lambda_c
        Npl = ((1.0 - np.exp(-(lambda_0 / lambda_c) ** beta)) *
               (C_NM / lambda_c) ** 3.0 /
               (np.exp(H_CGS * C_NM / (lambda_c * K_CGS * temperature)) - 1.0))

        # Conversion factor from F_lambda to F_nu
        # F_nu = F_lambda * lambda^2 / c
        conv = C_NM / (wave_nm * wave_nm)

        # Modified blackbody component (graybody)
        # B_lambda * (1 - exp(-(lambda_0/lambda)^beta))
        # where B_lambda is Planck function
        with np.errstate(over='ignore', divide='ignore'):
            exponent = H_CGS * C_NM / (wave_nm * K_CGS * temperature)
            # Avoid overflow for very short wavelengths
            exponent = np.clip(exponent, 0, 700)
            blackbody = (conv * (1.0 - np.exp(-(lambda_0 / wave_nm) ** beta)) *
                        (C_NM / wave_nm) ** 3.0 /
                        (np.exp(exponent) - 1.0))

        # Mid-IR power-law component
        # Gaussian cutoff at lambda_c prevents it from dominating at long wavelengths
        powerlaw = (conv * Npl * (wave_nm / lambda_c) ** alpha *
                   np.exp(-(wave_nm / lambda_c) ** 2.0))

        # Handle any NaN or Inf values
        blackbody = np.nan_to_num(blackbody, nan=0.0, posinf=0.0, neginf=0.0)
        powerlaw = np.nan_to_num(powerlaw, nan=0.0, posinf=0.0, neginf=0.0)

        # Total spectrum (in arbitrary units of W/nm initially)
        lumin_total = powerlaw + blackbody

        # Convert from W/nm to L_sun/Hz before normalizing
        # F_nu (W/Hz) = F_lambda (W/nm) * lambda^2 / c_nm
        # Then divide by L_sun to get L_sun/Hz
        lumin_Lsun_Hz = lumin_total * (wave_nm ** 2 / C_NM) / L_SUN

        # Normalize so integral over frequency = 1 L_sun
        # Convert wavelength to frequency for integration
        nu = C_NM / wave_nm  # Hz (decreasing with increasing wavelength)
        norm = -np.trapz(lumin_Lsun_Hz, nu)  # Negative because nu decreases

        if norm > 0:
            lumin_Lsun_Hz /= norm

        # Convert wavelength to Angstroms
        wave_aa = wave_nm * 10.0

        return wave_aa, lumin_Lsun_Hz

    def get_spectrum_components(self, temperature, beta, alpha, target_wave=None,
                                 wave_min=1e3, wave_max=1e6, n_wave=1000):
        """
        Compute the Casey2012 spectrum with separate components.

        Returns both the power-law and blackbody components separately,
        useful for diagnostics.

        Parameters
        ----------
        temperature : float
            Dust temperature in Kelvin
        beta : float
            Dust emissivity index
        alpha : float
            Mid-IR power-law slope
        target_wave : ndarray, optional
            Target wavelength grid in Angstroms
        wave_min, wave_max, n_wave : float, float, int
            Default wavelength grid parameters (in nm)

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec_total : ndarray
            Total spectrum in L_sun/Hz
        spec_powerlaw : ndarray
            Power-law component in L_sun/Hz
        spec_blackbody : ndarray
            Modified blackbody component in L_sun/Hz
        """
        # Create wavelength grid in nm
        if target_wave is not None:
            wave_nm = target_wave / 10.0
        else:
            wave_nm = np.logspace(np.log10(wave_min), np.log10(wave_max), n_wave)

        # Model constants
        b1 = 26.68
        b2 = 6.246
        b3 = 1.905e-4
        b4 = 7.243e-5

        lambda_c = 0.75e3 / ((b1 + b2 * alpha) ** -2.0 + (b3 + b4 * alpha) * temperature)
        lambda_0 = 200e3

        Npl = ((1.0 - np.exp(-(lambda_0 / lambda_c) ** beta)) *
               (C_NM / lambda_c) ** 3.0 /
               (np.exp(H_CGS * C_NM / (lambda_c * K_CGS * temperature)) - 1.0))

        conv = C_NM / (wave_nm * wave_nm)

        with np.errstate(over='ignore', divide='ignore'):
            exponent = H_CGS * C_NM / (wave_nm * K_CGS * temperature)
            exponent = np.clip(exponent, 0, 700)
            blackbody = (conv * (1.0 - np.exp(-(lambda_0 / wave_nm) ** beta)) *
                        (C_NM / wave_nm) ** 3.0 /
                        (np.exp(exponent) - 1.0))

        powerlaw = (conv * Npl * (wave_nm / lambda_c) ** alpha *
                   np.exp(-(wave_nm / lambda_c) ** 2.0))

        blackbody = np.nan_to_num(blackbody, nan=0.0, posinf=0.0, neginf=0.0)
        powerlaw = np.nan_to_num(powerlaw, nan=0.0, posinf=0.0, neginf=0.0)

        lumin_total = powerlaw + blackbody

        # Convert to L_sun/Hz before normalizing
        lumin_total_Lsun = lumin_total * (wave_nm ** 2 / C_NM) / L_SUN
        lumin_pl_Lsun = powerlaw * (wave_nm ** 2 / C_NM) / L_SUN
        lumin_bb_Lsun = blackbody * (wave_nm ** 2 / C_NM) / L_SUN

        # Normalize so integral over frequency = 1 L_sun
        nu = C_NM / wave_nm
        norm = -np.trapz(lumin_total_Lsun, nu)

        if norm > 0:
            lumin_total_Lsun /= norm
            lumin_pl_Lsun /= norm
            lumin_bb_Lsun /= norm

        wave_aa = wave_nm * 10.0

        return wave_aa, lumin_total_Lsun, lumin_pl_Lsun, lumin_bb_Lsun
