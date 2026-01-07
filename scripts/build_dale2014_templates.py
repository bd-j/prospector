#!/usr/bin/env python
"""
Build Dale2014 template file for Prospector from CIGALE raw data.

This script converts the Dale et al. (2014) dust emission templates from
CIGALE format to a compact NPZ file that Prospector can load efficiently.

The Dale2014 model is a single-parameter dust emission model where the
dust heating intensity distribution follows dM_d(U) ∝ U^(-alpha) dU.

Reference:
    Dale, D.A., et al. (2014), ApJ, 784, 83

Usage:
    python build_dale2014_templates.py /path/to/cigale/database_builder/dale2014/data [output.npz]
"""

import os
import sys
import numpy as np

# Physical constants
L_SUN = 3.846e26  # Solar luminosity in W


def build_templates(cigale_data_path, output_path):
    """
    Convert CIGALE Dale2014 raw templates to Prospector format.

    Parameters
    ----------
    cigale_data_path : str
        Path to CIGALE's database_builder/dale2014/data directory
    output_path : str
        Path for output NPZ file
    """
    # Read alpha grid
    dhcal_file = os.path.join(cigale_data_path, "dhcal.dat")
    if not os.path.exists(dhcal_file):
        raise FileNotFoundError(f"Cannot find alpha grid file: {dhcal_file}")

    d14cal = np.genfromtxt(dhcal_file)
    alpha_values = d14cal[:, 1]
    n_alpha = len(alpha_values)
    print(f"Found {n_alpha} alpha values: {alpha_values.min():.4f} to {alpha_values.max():.4f}")

    # Read main template file
    spectra_file = os.path.join(cigale_data_path, "spectra.0.00AGN.dat")
    if not os.path.exists(spectra_file):
        raise FileNotFoundError(f"Cannot find spectra file: {spectra_file}")

    # First column is wavelength in microns, rest are log10(nuFnu) for each alpha
    data = np.genfromtxt(spectra_file)
    wave_um = data[:, 0]  # microns
    wave_nm = wave_um * 1e3  # nm
    wave_aa = wave_um * 1e4  # Angstroms
    n_wave = len(wave_aa)

    print(f"Wavelength grid: {wave_um.min():.3f} - {wave_um.max():.1f} microns ({n_wave} points)")

    # Read stellar emission to subtract
    stell_file = os.path.join(cigale_data_path, "stellar_SED_age13Gyr_tau10Gyr.spec")
    if not os.path.exists(stell_file):
        raise FileNotFoundError(f"Cannot find stellar file: {stell_file}")

    stell_data = np.genfromtxt(stell_file)
    wave_stell_aa = stell_data[:, 0]  # Angstroms
    wave_stell_nm = wave_stell_aa * 0.1  # nm
    # Stellar emission in W/A -> W/nm
    stell_emission = stell_data[:, 1] * 10  # W/nm

    # Interpolate stellar emission to template wavelength grid
    stell_emission_interp = np.interp(wave_nm, wave_stell_nm, stell_emission)

    # Process templates
    # Data columns 1-64 are log10(nuFnu) for each alpha value
    # We need to:
    # 1. Convert from log10(nuFnu) to linear nuFnu
    # 2. Convert nuFnu to Fnu by dividing by nu (or equivalently, multiply by lambda/c and divide by lambda -> divide by c)
    #    Actually: nuFnu -> Fnu = nuFnu / nu, but we want F_lambda = Fnu * |dnu/dlambda| = Fnu * c/lambda^2
    #    From nuFnu: F_lambda = nuFnu / lambda (since nu*F_nu = lambda*F_lambda)
    # 3. Subtract stellar emission
    # 4. Normalize to 1W emitted

    templates = np.zeros((n_alpha, n_wave))

    for i, alpha in enumerate(alpha_values):
        # Get log10(nuFnu) for this alpha (column i+1 since column 0 is wavelength)
        log_nuFnu = data[:, i + 1]

        # Convert to linear and then to W/nm
        # nuFnu -> F_lambda = nuFnu / lambda (where lambda in nm gives W/nm)
        nuFnu = 10**log_nuFnu
        lumin_with_stell = nuFnu / wave_nm  # W/nm

        # Find normalization constant using a reference wavelength (around 2um = 2000nm)
        ref_idx = 7  # Index near 2um in the template grid
        constant = lumin_with_stell[ref_idx] / stell_emission_interp[ref_idx]

        # Subtract stellar emission
        lumin = lumin_with_stell - stell_emission_interp * constant

        # Zero out negative values and shortward of 2um (2000nm = 20000AA)
        lumin[lumin < 0] = 0
        lumin[wave_nm < 2e3] = 0

        # Normalize to 1W emitted
        norm = np.trapz(lumin, x=wave_nm)
        if norm > 0:
            lumin /= norm

        # Convert from W/nm to L_sun/Hz for consistency with DL2014
        # F_lambda (W/nm) -> F_nu (W/Hz) = F_lambda * lambda^2 / c
        # Then divide by L_sun to get L_sun/Hz
        c_nm = 2.998e17  # nm/s
        lumin_Lsun_Hz = lumin * (wave_nm**2 / c_nm) / L_SUN

        templates[i, :] = lumin_Lsun_Hz

    # Save to NPZ
    np.savez_compressed(
        output_path,
        wavelength=wave_aa,
        alpha_values=alpha_values,
        templates=templates
    )

    print(f"\nTemplates saved to {output_path}")
    print(f"  Wavelength: {wave_aa.min():.1f} - {wave_aa.max():.1f} Angstrom ({n_wave} points)")
    print(f"  Alpha values: {alpha_values.min():.4f} - {alpha_values.max():.4f} ({n_alpha} values)")
    print(f"  File size: {os.path.getsize(output_path) / 1e3:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_dale2014_templates.py /path/to/cigale/dale2014/data [output.npz]")
        print("\nExample:")
        print("  python build_dale2014_templates.py /Users/joe/python/cigale/database_builder/dale2014/data")
        sys.exit(1)

    cigale_path = sys.argv[1]

    # Default output location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(
        script_dir, "..", "prospect", "sources", "dust_data", "dale2014", "templates.npz"
    )

    output = sys.argv[2] if len(sys.argv) > 2 else default_output

    build_templates(cigale_path, output)
