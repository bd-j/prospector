#!/usr/bin/env python
"""
Build DL2014 template file for Prospector from CIGALE raw data.

This script converts the DL2014 dust emission templates from CIGALE format
to a compact NPZ file that Prospector can load efficiently.

Usage:
    python build_dl2014_templates.py /path/to/cigale/database_builder/dl2014/data [output.npz]

Template Format:
    Input (CIGALE):
        - Wavelength in micrometers (decreasing order)
        - Flux in Jy cm^2 sr^-1 H^-1 (column 2 = total)

    Output (Prospector):
        - Wavelength in Angstroms (increasing order)
        - Flux in L_sun/Hz per kg of dust
"""

import os
import sys
import io
import numpy as np

# Physical constants
C_CGS = 2.998e10  # cm/s
C_AA = 2.998e18   # Angstrom/s
M_H = 1.6737e-27  # hydrogen mass in kg
L_SUN = 3.846e26  # Solar luminosity in W


def build_templates(cigale_data_path, output_path):
    """
    Convert CIGALE DL2014 raw templates to Prospector format.

    Parameters
    ----------
    cigale_data_path : str
        Path to CIGALE's database_builder/dl2014/data directory
    output_path : str
        Path for output NPZ file
    """
    # qPAH code mapping (from CIGALE builder)
    qpah_map = {
        "000": 0.47, "010": 1.12, "020": 1.77, "030": 2.50,
        "040": 3.19, "050": 3.90, "060": 4.58, "070": 5.26,
        "080": 5.95, "090": 6.63, "100": 7.32
    }

    # Umin values (from CIGALE builder)
    umin_list = [
        "0.100", "0.120", "0.150", "0.170", "0.200", "0.250",
        "0.300", "0.350", "0.400", "0.500", "0.600", "0.700",
        "0.800", "1.000", "1.200", "1.500", "1.700", "2.000",
        "2.500", "3.000", "3.500", "4.000", "5.000", "6.000",
        "7.000", "8.000", "10.00", "12.00", "15.00", "17.00",
        "20.00", "25.00", "30.00", "35.00", "40.00", "50.00"
    ]

    # Alpha values (from CIGALE builder)
    alpha_list = [f"{a:.1f}" for a in np.arange(1.0, 3.05, 0.1)]

    # Mdust/MH ratio for unit conversion
    MdMH = {
        "000": 0.0100, "010": 0.0100, "020": 0.0101, "030": 0.0102,
        "040": 0.0102, "050": 0.0103, "060": 0.0104, "070": 0.0105,
        "080": 0.0106, "090": 0.0107, "100": 0.0108
    }

    # Read wavelength grid from first file
    first_file = os.path.join(cigale_data_path, "U0.100_0.100_MW3.1_000", "spec_1.0.dat")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"Cannot find template file: {first_file}")

    with open(first_file) as f:
        data = "".join(f.readlines()[-1001:])
    wave_um = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))[::-1]
    wave_aa = wave_um * 1e4  # Convert um to Angstrom
    wave_nm = wave_um * 1e3  # nm for unit conversion

    # Conversion factor from Jy cm^2 sr^-1 H^-1 to W/nm per kg of dust
    # Following CIGALE's database builder:
    # conv = 4*pi * 1e-30 / m_H * c / wave_nm^2 * 1e9
    # where 1e-30 converts Jy cm^2 to W/Hz/sr and 1e9 converts to nm
    conv_W_nm_per_kg = 4.0 * np.pi * 1e-30 / M_H * C_CGS / (wave_nm ** 2) * 1e9

    # Further convert W/nm to L_sun/Hz
    # L_nu = L_lambda * lambda^2 / c
    # L_sun/Hz = (W/nm / L_sun) * (wave_nm^2 / c_nm)
    c_nm = C_CGS * 1e7  # nm/s
    conv_Lsun_Hz = conv_W_nm_per_kg * (wave_nm ** 2 / c_nm) / L_SUN

    # Initialize arrays
    n_qpah = len(qpah_map)
    n_umin = len(umin_list)
    n_alpha = len(alpha_list)
    n_wave = len(wave_aa)

    qpah_values = np.array(sorted(qpah_map.values()))
    umin_values = np.array([float(u) for u in umin_list])
    alpha_values = np.array([float(a) for a in alpha_list])

    templates_minmin = np.zeros((n_qpah, n_umin, n_wave))
    templates_minmax = np.zeros((n_qpah, n_umin, n_alpha, n_wave))

    qpah_to_idx = {v: i for i, v in enumerate(sorted(qpah_map.values()))}

    print(f"Building templates from {cigale_data_path}")
    print(f"  {n_qpah} qpah values x {n_umin} umin values x {n_alpha} alpha values")

    total = n_qpah * n_umin * (1 + n_alpha)
    count = 0

    for model_code, qpah_val in qpah_map.items():
        qpah_idx = qpah_to_idx[qpah_val]

        for umin_idx, umin in enumerate(umin_list):
            # Delta function (U = Umin = Umax)
            dirname = f"U{umin}_{umin}_MW3.1_{model_code}"
            filepath = os.path.join(cigale_data_path, dirname, "spec_1.0.dat")

            with open(filepath) as f:
                data = "".join(f.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))[::-1]
            lumin *= conv_Lsun_Hz / MdMH[model_code]
            templates_minmin[qpah_idx, umin_idx, :] = lumin
            count += 1

            # Power-law distribution (Umin to Umax=1e7)
            for alpha_idx, alpha in enumerate(alpha_list):
                dirname = f"U{umin}_1e7_MW3.1_{model_code}"
                filepath = os.path.join(cigale_data_path, dirname, f"spec_{alpha}.dat")

                with open(filepath) as f:
                    data = "".join(f.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))[::-1]
                lumin *= conv_Lsun_Hz / MdMH[model_code]
                templates_minmax[qpah_idx, umin_idx, alpha_idx, :] = lumin
                count += 1

            if count % 100 == 0:
                print(f"  Processed {count}/{total} templates...")

    # Save to NPZ
    np.savez_compressed(
        output_path,
        wavelength=wave_aa,
        qpah_values=qpah_values,
        umin_values=umin_values,
        alpha_values=alpha_values,
        templates_minmin=templates_minmin,
        templates_minmax=templates_minmax
    )

    print(f"\nTemplates saved to {output_path}")
    print(f"  Wavelength: {wave_aa.min():.1f} - {wave_aa.max():.1f} Angstrom ({n_wave} points)")
    print(f"  qPAH values: {qpah_values}")
    print(f"  Umin range: {umin_values.min()} - {umin_values.max()} ({n_umin} values)")
    print(f"  Alpha range: {alpha_values.min()} - {alpha_values.max()} ({n_alpha} values)")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_dl2014_templates.py /path/to/cigale/dl2014/data [output.npz]")
        print("\nExample:")
        print("  python build_dl2014_templates.py /Users/joe/python/cigale/database_builder/dl2014/data")
        sys.exit(1)

    cigale_path = sys.argv[1]

    # Default output location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(
        script_dir, "..", "prospect", "sources", "dust_data", "dl2014", "templates.npz"
    )

    output = sys.argv[2] if len(sys.argv) > 2 else default_output

    build_templates(cigale_path, output)
