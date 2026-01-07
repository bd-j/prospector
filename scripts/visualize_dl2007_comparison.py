#!/usr/bin/env python
"""
Visualize and compare FSPS DL2007 vs CIGALE DL2007 dust emission templates.

This script plots dust emission spectra side-by-side for different parameter
combinations to help identify differences between the two implementations.

Usage:
    python visualize_dl2007_comparison.py

Requires: matplotlib, numpy, fsps
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

# Import CIGALE DL2007 templates
exec(open(os.path.join(project_dir, 'prospect/sources/dl2007.py')).read())

# Try to import FSPS
try:
    import fsps
    HAS_FSPS = True
except ImportError:
    print("Warning: FSPS not available. Only showing CIGALE templates.")
    HAS_FSPS = False


def get_fsps_dust_emission(wave, qpah, umin, gamma):
    """Get FSPS DL2007 dust emission spectrum."""
    if not HAS_FSPS:
        return None

    # Create SSP with dust emission
    ssp = fsps.StellarPopulation(
        zcontinuous=1,
        add_dust_emission=True,
        duste_qpah=qpah,
        duste_umin=umin,
        duste_gamma=gamma,
    )
    ssp.params['logzsol'] = 0.0
    ssp.params['dust_type'] = 0
    ssp.params['dust2'] = 0.3
    ssp.params['sfh'] = 0

    wave_fsps, spec_with = ssp.get_spectrum(tage=10.0, peraa=False)

    # Get spectrum without dust emission
    ssp.params['add_dust_emission'] = False
    _, spec_no = ssp.get_spectrum(tage=10.0, peraa=False)

    # Dust emission = with - without
    dust_fsps = spec_with - spec_no

    # Interpolate to requested wavelength grid
    dust_interp = np.interp(wave, wave_fsps, dust_fsps, left=0, right=0)

    return dust_interp


def get_cigale_dust_emission(wave, qpah, umin, umax, gamma):
    """Get CIGALE DL2007 dust emission spectrum (normalized to 1 L_sun)."""
    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/dl2007/templates.npz')
    DL2007Templates._instance = None
    dl2007 = DL2007Templates(template_path)

    _, spec_norm, emissivity = dl2007.get_template(qpah, umin, umax, gamma, target_wave=wave)
    return spec_norm


def plot_comparison_grid():
    """Plot comparison grid for different parameter combinations."""
    # Common wavelength grid (in Angstroms)
    wave = np.logspace(np.log10(1e4), np.log10(1e8), 500)  # 1 um to 10 mm
    wave_um = wave / 1e4  # Convert to microns for plotting

    # Parameter combinations to test
    qpah_values = [0.47, 2.5, 4.58]
    umin_values = [0.1, 1.0, 10.0]
    gamma_values = [0.01, 0.1, 0.5]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('FSPS DL2007 vs CIGALE DL2007 Dust Emission Comparison', fontsize=14)

    # Fixed parameters for each row
    # Row 0: vary qpah (umin=1, gamma=0.1)
    # Row 1: vary umin (qpah=2.5, gamma=0.1)
    # Row 2: vary gamma (qpah=2.5, umin=1)

    umax = 1e6  # CIGALE umax

    for col, qpah in enumerate(qpah_values):
        ax = axes[0, col]
        umin, gamma = 1.0, 0.1

        cigale = get_cigale_dust_emission(wave, qpah, umin, umax, gamma)
        ax.loglog(wave_um, cigale * wave, 'b-', label='CIGALE', linewidth=2)

        if HAS_FSPS:
            fsps_dust = get_fsps_dust_emission(wave, qpah, umin, gamma)
            if fsps_dust is not None:
                # Normalize FSPS to match CIGALE integral for shape comparison
                C_AA = 2.998e18
                nu = C_AA / wave
                cigale_int = -np.trapz(cigale, nu)
                fsps_int = -np.trapz(fsps_dust, nu)
                if fsps_int > 0:
                    fsps_norm = fsps_dust / fsps_int * cigale_int
                    ax.loglog(wave_um, fsps_norm * wave, 'r--', label='FSPS (norm)', linewidth=2)

        ax.set_title(f'qpah={qpah}')
        ax.set_xlim(1, 1000)
        ax.set_ylabel('λ F_λ (arb units)' if col == 0 else '')
        if col == 2:
            ax.legend(loc='upper right', fontsize=8)
        ax.text(0.05, 0.95, f'umin={umin}, γ={gamma}', transform=ax.transAxes,
                fontsize=8, va='top')

    for col, umin in enumerate(umin_values):
        ax = axes[1, col]
        qpah, gamma = 2.5, 0.1

        cigale = get_cigale_dust_emission(wave, qpah, umin, umax, gamma)
        ax.loglog(wave_um, cigale * wave, 'b-', label='CIGALE', linewidth=2)

        if HAS_FSPS:
            fsps_dust = get_fsps_dust_emission(wave, qpah, umin, gamma)
            if fsps_dust is not None:
                C_AA = 2.998e18
                nu = C_AA / wave
                cigale_int = -np.trapz(cigale, nu)
                fsps_int = -np.trapz(fsps_dust, nu)
                if fsps_int > 0:
                    fsps_norm = fsps_dust / fsps_int * cigale_int
                    ax.loglog(wave_um, fsps_norm * wave, 'r--', label='FSPS (norm)', linewidth=2)

        ax.set_title(f'umin={umin}')
        ax.set_xlim(1, 1000)
        ax.set_ylabel('λ F_λ (arb units)' if col == 0 else '')
        ax.text(0.05, 0.95, f'qpah={qpah}, γ={gamma}', transform=ax.transAxes,
                fontsize=8, va='top')

    for col, gamma in enumerate(gamma_values):
        ax = axes[2, col]
        qpah, umin = 2.5, 1.0

        cigale = get_cigale_dust_emission(wave, qpah, umin, umax, gamma)
        ax.loglog(wave_um, cigale * wave, 'b-', label='CIGALE', linewidth=2)

        if HAS_FSPS:
            fsps_dust = get_fsps_dust_emission(wave, qpah, umin, gamma)
            if fsps_dust is not None:
                C_AA = 2.998e18
                nu = C_AA / wave
                cigale_int = -np.trapz(cigale, nu)
                fsps_int = -np.trapz(fsps_dust, nu)
                if fsps_int > 0:
                    fsps_norm = fsps_dust / fsps_int * cigale_int
                    ax.loglog(wave_um, fsps_norm * wave, 'r--', label='FSPS (norm)', linewidth=2)

        ax.set_title(f'gamma={gamma}')
        ax.set_xlim(1, 1000)
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('λ F_λ (arb units)' if col == 0 else '')
        ax.text(0.05, 0.95, f'qpah={qpah}, umin={umin}', transform=ax.transAxes,
                fontsize=8, va='top')

    plt.tight_layout()
    plt.show()


def plot_raw_templates():
    """Plot the raw CIGALE templates to verify they loaded correctly."""
    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/dl2007/templates.npz')
    DL2007Templates._instance = None
    dl2007 = DL2007Templates(template_path)

    print("DL2007 Template Info:")
    print(f"  Wavelength: {dl2007.wavelength.min():.0f} - {dl2007.wavelength.max():.0f} Angstrom")
    print(f"  qpah values: {dl2007.qpah_values}")
    print(f"  umin values: {dl2007.umin_values}")
    print(f"  umax values: {dl2007.umax_values}")

    wave_um = dl2007.wavelength / 1e4

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot minmin templates for different qpah at fixed umin
    ax = axes[0]
    umin_idx = np.argmin(np.abs(dl2007.umin_values - 1.0))
    for iq, qpah in enumerate(dl2007.qpah_values):
        spec = dl2007._templates_minmin[iq, umin_idx, :]
        ax.loglog(wave_um, spec * wave_um, label=f'qpah={qpah:.2f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (raw units)')
    ax.set_title(f'Raw minmin templates (umin={dl2007.umin_values[umin_idx]:.1f})')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Plot minmax templates for different alpha at fixed qpah, umin
    ax = axes[1]
    qpah_idx = 3  # middle qpah
    umax_idx = 3  # highest umax (1e6)
    for iu, umin in enumerate(dl2007.umin_values[::5]):  # every 5th umin
        iu_actual = iu * 5
        spec = dl2007._templates_minmax[qpah_idx, iu_actual, umax_idx, :]
        ax.loglog(wave_um, spec * wave_um, label=f'umin={umin:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (raw units)')
    ax.set_title(f'Raw minmax templates (qpah={dl2007.qpah_values[qpah_idx]:.2f}, umax={dl2007.umax_values[umax_idx]:.0e})')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("DL2007 Dust Emission Visualization")
    print("=" * 60)

    print("\n1. Plotting raw CIGALE templates...")
    plot_raw_templates()

    print("\n2. Plotting FSPS vs CIGALE comparison...")
    plot_comparison_grid()


if __name__ == "__main__":
    main()
