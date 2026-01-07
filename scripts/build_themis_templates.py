#!/usr/bin/env python
"""
Build Themis (Jones et al. 2017) dust emission templates from CIGALE raw data.

This script reads the Themis raw template files from CIGALE and converts them
to a single NPZ file for use in Prospector.

The raw templates are in units of Jy cm² sr⁻¹ H⁻¹ and need to be converted
to L_sun/Hz (per unit mass).

Themis model parameters:
- qhac: Mass fraction of hydrocarbon solids (HAC) [0.02 - 0.40, 11 values]
- umin: Minimum radiation field intensity [0.10 - 80.0, 37 values]
- alpha: Power-law slope for dU/dM distribution [1.0 - 3.0, 21 values]
- gamma: Fraction of dust in PDR component [0 - 1]
- umax: Fixed at 1e7

Reference:
    Jones, A.P. et al. (2017), A&A, 602, A46 (THEMIS model)

Usage:
    python build_themis_templates.py /path/to/cigale/database_builder/themis/data

"""

import io
import os
import sys
from pathlib import Path
import numpy as np
import scipy.constants as cst

# Physical constants for unit conversion
L_SUN = 3.846e26  # Solar luminosity in W
C_NM = 2.998e17   # Speed of light in nm/s


def build_themis_templates(cigale_data_path, output_path):
    """
    Build Themis templates from CIGALE raw data files.

    Parameters
    ----------
    cigale_data_path : str
        Path to CIGALE's database_builder/themis/data directory
    output_path : str
        Path for output NPZ file
    """
    path = Path(cigale_data_path)

    # Model parameters following CIGALE's database builder
    qhac_dict = {
        "000": 0.02,
        "010": 0.06,
        "020": 0.10,
        "030": 0.14,
        "040": 0.17,
        "050": 0.20,
        "060": 0.24,
        "070": 0.28,
        "080": 0.32,
        "090": 0.36,
        "100": 0.40,
    }

    umin_values = [
        "0.100", "0.120", "0.150", "0.170", "0.200", "0.250", "0.300", "0.350",
        "0.400", "0.500", "0.600", "0.700", "0.800", "1.000", "1.200", "1.500",
        "1.700", "2.000", "2.500", "3.000", "3.500", "4.000", "5.000", "6.000",
        "7.000", "8.000", "10.00", "12.00", "15.00", "17.00", "20.00", "25.00",
        "30.00", "35.00", "40.00", "50.00", "80.00"
    ]

    alpha_values = [
        "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9",
        "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3.0"
    ]

    # Mdust/MH ratio (constant for all qhac models in Themis)
    MdMH = 7.4e-3

    # Read wavelength grid from a reference file
    ref_file = path / "U0.100_0.100_MW3.1_000" / "spec_1.0.dat"
    print(f"Reading wavelength grid from {ref_file}")
    with open(ref_file) as f:
        data = "".join(f.readlines()[-576:])
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # Convert from μm to nm
    wave_nm = wave * 1000.0
    n_wave = len(wave_nm)

    # Conversion factor from Jy cm² sr⁻¹ H⁻¹ to W nm⁻¹ (kg of H)⁻¹
    # Following CIGALE's exact formula
    conv = 4.0 * np.pi * 1e-30 / (cst.m_p + cst.m_e) * cst.c / (wave_nm * wave_nm) * 1e9

    # Prepare output arrays
    qhac_arr = np.array(list(qhac_dict.values()))
    umin_arr = np.array([float(u) for u in umin_values])
    alpha_arr = np.array([float(a) for a in alpha_values])

    n_qhac = len(qhac_arr)
    n_umin = len(umin_arr)
    n_alpha = len(alpha_arr)

    # Templates array:
    # minmin: (qhac, umin, wave) - delta function at umin
    # minmax: (qhac, umin, alpha, wave) - power-law from umin to umax=1e7
    templates_minmin = np.zeros((n_qhac, n_umin, n_wave), dtype=np.float32)
    templates_minmax = np.zeros((n_qhac, n_umin, n_alpha, n_wave), dtype=np.float32)

    print(f"Building templates for {n_qhac} qhac x {n_umin} umin values")
    print(f"Plus {n_alpha} alpha values for the power-law component")

    model_codes = list(qhac_dict.keys())

    for iq, model_code in enumerate(model_codes):
        qhac = qhac_dict[model_code]
        print(f"  Processing qhac={qhac:.2f} (model {model_code})")

        for iu, umin in enumerate(umin_values):
            # Read delta function model (umin = umax)
            filename = path / f"U{umin}_{umin}_MW3.1_{model_code}" / "spec_1.0.dat"
            with open(filename) as f:
                data = "".join(f.readlines()[-576:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # Convert from Jy cm² sr⁻¹ H⁻¹ to W nm⁻¹ (kg of dust)⁻¹
            lumin = lumin * conv / MdMH
            templates_minmin[iq, iu, :] = lumin

            # Read power-law models (umin to umax=1e7)
            for ia, alpha in enumerate(alpha_values):
                filename = path / f"U{umin}_1e7_MW3.1_{model_code}" / f"spec_{alpha}.dat"
                with open(filename) as f:
                    data = "".join(f.readlines()[-576:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                lumin = lumin * conv / MdMH
                templates_minmax[iq, iu, ia, :] = lumin

    # Convert wavelength to Angstroms for Prospector
    wave_aa = wave_nm * 10.0

    # Save templates
    print(f"Saving templates to {output_path}")
    np.savez_compressed(
        output_path,
        wavelength=wave_aa,
        qhac_values=qhac_arr,
        umin_values=umin_arr,
        alpha_values=alpha_arr,
        templates_minmin=templates_minmin,
        templates_minmax=templates_minmax,
    )

    # Report file size
    file_size = os.path.getsize(output_path) / 1e6
    print(f"Template file size: {file_size:.1f} MB")

    return wave_aa, qhac_arr, umin_arr, alpha_arr


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_themis_templates.py /path/to/cigale/database_builder/themis/data [output_path]")
        print("")
        print("Example:")
        print("  python build_themis_templates.py ~/cigale/database_builder/themis/data")
        sys.exit(1)

    cigale_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        # Default output path
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "prospect" / "sources" / "dust_data" / "themis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "templates.npz")

    wave, qhac, umin, alpha = build_themis_templates(cigale_path, output_path)

    print("\nTemplate summary:")
    print(f"  Wavelength range: {wave.min():.0f} - {wave.max():.0f} Angstroms")
    print(f"  qhac values ({len(qhac)}): {qhac}")
    print(f"  umin values ({len(umin)}): {umin.min():.2f} - {umin.max():.1f}")
    print(f"  alpha values ({len(alpha)}): {alpha.min():.1f} - {alpha.max():.1f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
