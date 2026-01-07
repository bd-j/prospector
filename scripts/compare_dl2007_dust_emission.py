#!/usr/bin/env python
"""
Direct comparison of FSPS DL2007 vs CIGALE DL2007 dust emission templates.

This script isolates JUST the dust emission component to verify that
the CIGALE-ported templates match FSPS's built-in DL2007 templates.

Approach:
1. Generate a stellar spectrum with FSPS (with dust attenuation but NO dust emission)
2. Generate the same spectrum WITH dust emission
3. Subtract to isolate the dust emission component from FSPS
4. Compare with CIGALE DL2007 templates scaled by the same absorbed energy

Usage:
    python compare_dl2007_dust_emission.py
"""

import numpy as np
import sys
import os

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

try:
    import fsps
    HAS_FSPS = True
except ImportError:
    print("FSPS not available - cannot run comparison")
    HAS_FSPS = False
    sys.exit(1)

# Import CIGALE DL2007 templates
exec(open(os.path.join(project_dir, 'prospect/sources/dl2007.py')).read())


def get_fsps_dust_emission(qpah, umin, gamma, tage=10.0, logzsol=0.0, dust2=0.3):
    """
    Extract just the dust emission component from FSPS.

    Returns the dust emission spectrum and the absorbed luminosity.
    """
    # Create SSP with dust emission ON
    ssp_with_dust = fsps.StellarPopulation(
        zcontinuous=1,
        add_dust_emission=True,
        duste_qpah=qpah,
        duste_umin=umin,
        duste_gamma=gamma,
    )
    ssp_with_dust.params['logzsol'] = logzsol
    ssp_with_dust.params['dust_type'] = 0  # Power law
    ssp_with_dust.params['dust2'] = dust2
    ssp_with_dust.params['sfh'] = 0  # SSP

    wave, spec_with_dust = ssp_with_dust.get_spectrum(tage=tage, peraa=False)

    # Create SSP with dust emission OFF (but same attenuation)
    ssp_no_dust = fsps.StellarPopulation(
        zcontinuous=1,
        add_dust_emission=False,
    )
    ssp_no_dust.params['logzsol'] = logzsol
    ssp_no_dust.params['dust_type'] = 0
    ssp_no_dust.params['dust2'] = dust2
    ssp_no_dust.params['sfh'] = 0

    wave2, spec_no_dust = ssp_no_dust.get_spectrum(tage=tage, peraa=False)

    # Also get intrinsic (no dust at all) to calculate absorbed energy
    ssp_intrinsic = fsps.StellarPopulation(
        zcontinuous=1,
        add_dust_emission=False,
    )
    ssp_intrinsic.params['logzsol'] = logzsol
    ssp_intrinsic.params['dust_type'] = 0
    ssp_intrinsic.params['dust2'] = 0.0  # No attenuation
    ssp_intrinsic.params['sfh'] = 0

    wave3, spec_intrinsic = ssp_intrinsic.get_spectrum(tage=tage, peraa=False)

    # Dust emission = spectrum with dust - spectrum without dust emission
    dust_emission_fsps = spec_with_dust - spec_no_dust

    # Calculate absorbed luminosity
    C_AA = 2.998e18
    nu = C_AA / wave
    L_absorbed = -np.trapz(spec_intrinsic - spec_no_dust, nu)

    # Total dust luminosity from FSPS
    L_dust_fsps = -np.trapz(dust_emission_fsps, nu)

    return wave, dust_emission_fsps, spec_no_dust, L_absorbed, L_dust_fsps


def get_cigale_dust_emission(wave, qpah, umin, umax, gamma, L_absorbed):
    """
    Get CIGALE DL2007 dust emission scaled by absorbed luminosity.
    """
    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/dl2007/templates.npz')
    dl2007 = DL2007Templates(template_path)

    # Get template normalized to 1 L_sun
    _, dust_spec_norm, emissivity = dl2007.get_template(qpah, umin, umax, gamma, target_wave=wave)

    # Scale by absorbed luminosity
    dust_emission_cigale = L_absorbed * dust_spec_norm

    # Total dust luminosity
    C_AA = 2.998e18
    nu = C_AA / wave
    L_dust_cigale = -np.trapz(dust_emission_cigale, nu)

    return dust_emission_cigale, L_dust_cigale


def compare_dust_emission():
    """
    Compare FSPS and CIGALE DL2007 dust emission directly.
    """
    print("=" * 70)
    print("Direct Comparison: FSPS DL2007 vs CIGALE DL2007 Dust Emission")
    print("=" * 70)

    # Test parameters - use values available in both implementations
    # CIGALE qpah: [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58]
    # FSPS qpah: 0.1 - 10.0 (continuous)
    qpah = 2.50
    umin = 1.0
    gamma = 0.1
    umax = 1e6  # CIGALE supports this; FSPS may have different default

    # Stellar parameters
    tage = 10.0  # Gyr
    logzsol = 0.0
    dust2 = 0.3

    print(f"\nParameters:")
    print(f"  qpah = {qpah}")
    print(f"  umin = {umin}")
    print(f"  gamma = {gamma}")
    print(f"  umax = {umax} (CIGALE)")
    print(f"  tage = {tage} Gyr")
    print(f"  dust2 = {dust2}")

    # Get FSPS dust emission
    print("\n--- Getting FSPS DL2007 dust emission ---")
    wave, dust_fsps, stellar_attenuated, L_abs_fsps, L_dust_fsps = get_fsps_dust_emission(
        qpah, umin, gamma, tage, logzsol, dust2
    )
    print(f"  Absorbed luminosity: {L_abs_fsps:.4e} L_sun/Msun")
    print(f"  Dust luminosity (FSPS): {L_dust_fsps:.4e} L_sun/Msun")
    print(f"  Energy balance ratio: {L_dust_fsps/L_abs_fsps:.4f} (should be ~1.0)")

    # Get CIGALE dust emission scaled by FSPS absorbed energy
    print("\n--- Getting CIGALE DL2007 dust emission ---")
    dust_cigale, L_dust_cigale = get_cigale_dust_emission(
        wave, qpah, umin, umax, gamma, L_abs_fsps
    )
    print(f"  Dust luminosity (CIGALE): {L_dust_cigale:.4e} L_sun/Msun")
    print(f"  Energy balance ratio: {L_dust_cigale/L_abs_fsps:.4f} (should be ~1.0)")

    # Compare
    print("\n--- Comparison ---")
    print(f"  L_dust ratio (CIGALE/FSPS): {L_dust_cigale/L_dust_fsps:.4f}")

    # Compare in different wavelength regions
    C_AA = 2.998e18
    regions = [
        ('Mid-IR (3-30 μm)', 30000, 300000),
        ('Far-IR (30-300 μm)', 300000, 3000000),
        ('Sub-mm (300-1000 μm)', 3000000, 10000000),
    ]

    print("\n  Ratio by wavelength region (CIGALE/FSPS):")
    for name, wmin, wmax in regions:
        mask = (wave >= wmin) & (wave <= wmax) & (dust_fsps > 0)
        if mask.sum() > 0:
            # Use flux-weighted mean ratio
            ratio = np.sum(dust_cigale[mask]) / np.sum(dust_fsps[mask])
            print(f"    {name}: {ratio:.4f}")

    # Find peak wavelengths
    ir_mask = wave > 30000  # > 3 microns
    peak_fsps = wave[ir_mask][np.argmax(dust_fsps[ir_mask])] / 1e4
    peak_cigale = wave[ir_mask][np.argmax(dust_cigale[ir_mask])] / 1e4
    print(f"\n  Peak wavelength (FSPS): {peak_fsps:.1f} μm")
    print(f"  Peak wavelength (CIGALE): {peak_cigale:.1f} μm")

    # Test different parameters
    print("\n" + "=" * 70)
    print("Parameter Sweep")
    print("=" * 70)

    # Test different qpah values
    print("\n--- qpah variation (umin=1.0, gamma=0.1) ---")
    qpah_values = [0.47, 1.77, 2.50, 3.90, 4.58]
    for qp in qpah_values:
        wave, dust_fsps, _, L_abs, L_dust_fsps = get_fsps_dust_emission(qp, 1.0, 0.1, tage, logzsol, dust2)
        dust_cigale, L_dust_cigale = get_cigale_dust_emission(wave, qp, 1.0, 1e6, 0.1, L_abs)
        ratio = L_dust_cigale / L_dust_fsps
        print(f"  qpah={qp:.2f}: L_ratio={ratio:.4f}")

    # Test different umin values
    print("\n--- umin variation (qpah=2.5, gamma=0.1) ---")
    umin_values = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0]
    for um in umin_values:
        wave, dust_fsps, _, L_abs, L_dust_fsps = get_fsps_dust_emission(2.5, um, 0.1, tage, logzsol, dust2)
        dust_cigale, L_dust_cigale = get_cigale_dust_emission(wave, 2.5, um, 1e6, 0.1, L_abs)
        ratio = L_dust_cigale / L_dust_fsps
        print(f"  umin={um:.1f}: L_ratio={ratio:.4f}")

    # Test different gamma values
    print("\n--- gamma variation (qpah=2.5, umin=1.0) ---")
    gamma_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    for gm in gamma_values:
        wave, dust_fsps, _, L_abs, L_dust_fsps = get_fsps_dust_emission(2.5, 1.0, gm, tage, logzsol, dust2)
        dust_cigale, L_dust_cigale = get_cigale_dust_emission(wave, 2.5, 1.0, 1e6, gm, L_abs)
        ratio = L_dust_cigale / L_dust_fsps
        print(f"  gamma={gm:.2f}: L_ratio={ratio:.4f}")

    return wave, dust_fsps, dust_cigale


def main():
    if not HAS_FSPS:
        print("FSPS is required for this comparison")
        sys.exit(1)

    compare_dust_emission()

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
