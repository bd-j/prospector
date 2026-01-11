#!/usr/bin/env python
"""
Build DL2007 (Draine & Li 2007) dust emission templates from CIGALE raw data.

This script reads the DL2007 raw template files from CIGALE and converts them
to a single NPZ file for use in Prospector.

The raw templates are in units of Jy cm² sr⁻¹ H⁻¹ and need to be converted
to L_sun/Hz (per unit mass).

Key differences from DL2014:
- DL2007 has variable umax (1e3, 1e4, 1e5, 1e6) instead of fixed 1e7
- DL2007 has alpha fixed at 2.0 (no alpha parameter)
- DL2007 has fewer qpah values (7 vs 11) and umin values (23 vs 36)

Reference:
    Draine, B.T. & Li, A. (2007), ApJ, 657, 810-837

Usage:
    python build_dl2007_templates.py /path/to/cigale/database_builder/dl2007/data

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


def build_dl2007_templates(cigale_data_path, output_path):
    """
    Build DL2007 templates from CIGALE raw data files.

    Parameters
    ----------
    cigale_data_path : str
        Path to CIGALE's database_builder/dl2007/data directory
    output_path : str
        Path for output NPZ file
    """
    path = Path(cigale_data_path)

    # Model parameters following CIGALE's database builder
    qpah_dict = {
        "00": 0.47,
        "10": 1.12,
        "20": 1.77,
        "30": 2.50,
        "40": 3.19,
        "50": 3.90,
        "60": 4.58,
    }

    umax_values = ["1e3", "1e4", "1e5", "1e6"]
    umin_values = [
        "0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.70", "0.80",
        "1.00", "1.20", "1.50", "2.00", "2.50", "3.00", "4.00", "5.00",
        "7.00", "8.00", "10.0", "12.0", "15.0", "20.0", "25.0"
    ]

    # Mdust/MH ratios for each qpah model
    MdMH = {
        "00": 0.0100,
        "10": 0.0100,
        "20": 0.0101,
        "30": 0.0102,
        "40": 0.0102,
        "50": 0.0103,
        "60": 0.0104,
    }

    # Read wavelength grid from a reference file
    ref_file = path / "U1e3" / "U1e3_1e3_MW3.1_00.txt"
    print(f"Reading wavelength grid from {ref_file}")
    with open(ref_file) as f:
        data = "".join(f.readlines()[-1001:])
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # Wavelengths are decreasing in model files, reverse to increasing
    wave = wave[::-1]
    # Convert from μm to nm
    wave_nm = wave * 1000.0
    n_wave = len(wave_nm)

    # Conversion factor from Jy cm² sr⁻¹ H⁻¹ to W nm⁻¹ (kg of H)⁻¹
    # Following CIGALE's exact formula
    conv = 4.0 * np.pi * 1e-30 / (cst.m_p + cst.m_e) * cst.c / (wave_nm * wave_nm) * 1e9

    # Prepare output arrays
    qpah_arr = np.array(list(qpah_dict.values()))
    umin_arr = np.array([float(u) for u in umin_values])
    umax_arr = np.array([float(u) for u in umax_values])

    n_qpah = len(qpah_arr)
    n_umin = len(umin_arr)
    n_umax = len(umax_arr)

    # Templates array: (qpah, umin, umax_type, wavelength)
    # umax_type: 0 = umin (delta function), 1-4 = 1e3, 1e4, 1e5, 1e6
    templates_minmin = np.zeros((n_qpah, n_umin, n_wave), dtype=np.float32)
    templates_minmax = np.zeros((n_qpah, n_umin, n_umax, n_wave), dtype=np.float32)

    print(f"Building templates for {n_qpah} qpah x {n_umin} umin values")
    print(f"Plus {n_umax} umax values for the power-law component")

    model_codes = list(qpah_dict.keys())

    for iq, model_code in enumerate(model_codes):
        qpah = qpah_dict[model_code]
        print(f"  Processing qpah={qpah:.2f} (model {model_code})")

        for iu, umin in enumerate(umin_values):
            # Read delta function model (umin = umax)
            filename = path / f"U{umin}" / f"U{umin}_{umin}_MW3.1_{model_code}.txt"
            with open(filename) as f:
                data = "".join(f.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            lumin = lumin[::-1]  # Reverse to match wavelength order
            # Convert from Jy cm² sr⁻¹ H⁻¹ to W nm⁻¹ (kg of dust)⁻¹
            lumin = lumin * conv / MdMH[model_code]
            templates_minmin[iq, iu, :] = lumin

            # Read power-law models (umin to umax)
            for iumax, umax in enumerate(umax_values):
                filename = path / f"U{umin}" / f"U{umin}_{umax}_MW3.1_{model_code}.txt"
                with open(filename) as f:
                    data = "".join(f.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                lumin = lumin[::-1]
                lumin = lumin * conv / MdMH[model_code]
                templates_minmax[iq, iu, iumax, :] = lumin

    # Convert wavelength to Angstroms for Prospector
    wave_aa = wave_nm * 10.0

    # Save templates
    print(f"Saving templates to {output_path}")
    np.savez_compressed(
        output_path,
        wavelength=wave_aa,
        qpah_values=qpah_arr,
        umin_values=umin_arr,
        umax_values=umax_arr,
        templates_minmin=templates_minmin,
        templates_minmax=templates_minmax,
    )

    # Report file size
    file_size = os.path.getsize(output_path) / 1e6
    print(f"Template file size: {file_size:.1f} MB")

    return wave_aa, qpah_arr, umin_arr, umax_arr


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_dl2007_templates.py /path/to/cigale/database_builder/dl2007/data [output_path]")
        print("")
        print("Example:")
        print("  python build_dl2007_templates.py ~/cigale/database_builder/dl2007/data")
        sys.exit(1)

    cigale_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        # Default output path
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "prospect" / "sources" / "dust_data" / "dl2007"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "templates.npz")

    wave, qpah, umin, umax = build_dl2007_templates(cigale_path, output_path)

    print("\nTemplate summary:")
    print(f"  Wavelength range: {wave.min():.0f} - {wave.max():.0f} Angstroms")
    print(f"  qpah values ({len(qpah)}): {qpah}")
    print(f"  umin values ({len(umin)}): {umin.min():.2f} - {umin.max():.1f}")
    print(f"  umax values ({len(umax)}): {umax}")
    print("\nDone!")


if __name__ == "__main__":
    main()
