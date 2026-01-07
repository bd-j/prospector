#!/usr/bin/env python
"""
Compare FSPS built-in DL2007 dust emission with CIGALE-ported DL2007.

This script generates galaxy spectra using both implementations with
matched parameters and compares the results, particularly in the IR
where dust emission dominates.

Usage:
    python compare_dl2007_implementations.py [--plot]

Requirements:
    - prospect with FSPS
    - numpy, matplotlib (for plotting)
"""

import argparse
import numpy as np
import sys

# Check if we can import prospect
try:
    from prospect.sources import CSPSpecBasis, DL2007CigaleSSPBasis
    from prospect.models.templates import TemplateLibrary
    HAS_PROSPECT = True
except ImportError as e:
    # Try alternative import path that doesn't require sedpy
    try:
        import os
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        sys.path.insert(0, project_dir)

        # Import sources directly without going through prospect/__init__.py
        from prospect.sources.galaxy_basis import CSPSpecBasis, FastStepBasis
        from prospect.sources.dl2007_basis import DL2007CigaleSSPBasis
        HAS_PROSPECT = True
        print("Using direct source imports (sedpy not available)")
    except ImportError as e2:
        print(f"Warning: Could not import prospect: {e}")
        print(f"Alternative import also failed: {e2}")
        print("Will test template loaders only.")
        HAS_PROSPECT = False


def compare_templates_only():
    """
    Compare just the dust emission templates without full SPS.
    This works even without FSPS installed.
    """
    print("=" * 70)
    print("Comparing DL2007 CIGALE Templates")
    print("=" * 70)

    # Import template loader directly
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    sys.path.insert(0, project_dir)
    exec(open(os.path.join(project_dir, 'prospect/sources/dl2007.py')).read(), globals())

    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/dl2007/templates.npz')
    dl2007 = DL2007Templates(template_path)

    print(f"\nTemplate grid:")
    print(f"  Wavelength: {dl2007.wavelength.min():.0f} - {dl2007.wavelength.max():.0f} Angstroms")
    print(f"  qpah values: {dl2007.qpah_values}")
    print(f"  umin values: {len(dl2007.umin_values)} values from {dl2007.umin_values.min():.2f} to {dl2007.umin_values.max():.1f}")
    print(f"  umax values: {dl2007.umax_values}")

    # Test different parameter combinations
    test_params = [
        {'qpah': 2.50, 'umin': 1.0, 'umax': 1e6, 'gamma': 0.1},
        {'qpah': 0.47, 'umin': 0.1, 'umax': 1e6, 'gamma': 0.01},
        {'qpah': 4.58, 'umin': 10.0, 'umax': 1e6, 'gamma': 0.5},
        {'qpah': 2.50, 'umin': 1.0, 'umax': 1e3, 'gamma': 0.1},  # Different umax
    ]

    print("\nTemplate tests:")
    for params in test_params:
        wave, spec, emiss = dl2007.get_template(**params)

        # Find peak wavelength
        peak_idx = np.argmax(spec)
        peak_wave = wave[peak_idx]
        peak_wave_um = peak_wave / 1e4  # Convert to microns

        # Check normalization
        C_AA = 2.998e18
        nu = C_AA / wave
        integral = -np.trapz(spec, nu)

        # Compute <U>
        umean = dl2007.compute_umean(params['umin'], params['gamma'], params['umax'])

        print(f"  qpah={params['qpah']:.2f}, umin={params['umin']:.1f}, "
              f"umax={params['umax']:.0e}, gamma={params['gamma']:.2f}:")
        print(f"    Peak at {peak_wave_um:.1f} μm, <U>={umean:.2f}, "
              f"norm={integral:.4f}")

    return dl2007


def compare_full_spectra(plot=False):
    """
    Compare full galaxy spectra using FSPS DL2007 vs CIGALE DL2007.
    Requires FSPS to be installed.
    """
    print("\n" + "=" * 70)
    print("Comparing Full Galaxy Spectra: FSPS DL2007 vs CIGALE DL2007")
    print("=" * 70)

    # DL2007 dust parameters - matched between implementations
    # FSPS uses: duste_qpah, duste_umin, duste_gamma
    # CIGALE uses: dl2007_cigale_qpah, dl2007_cigale_umin, dl2007_cigale_umax, dl2007_cigale_gamma

    qpah = 2.50
    umin = 1.0
    gamma = 0.1
    umax = 1e6  # CIGALE allows this; FSPS has it fixed

    print(f"\nDust parameters:")
    print(f"  qpah = {qpah}")
    print(f"  umin = {umin}")
    print(f"  gamma = {gamma}")
    print(f"  umax = {umax} (CIGALE only; FSPS may differ)")

    # Parameters for CSPSpecBasis (simple parametric SFH)
    # CSPSpecBasis expects 'tage' and 'mass' (scalar or array)
    fsps_base_params = {
        'zred': 0.1,
        'logzsol': 0.0,
        'dust2': 0.3,
        'dust_type': 0,  # Power-law attenuation
        'dust_index': -0.7,
        'tage': 10.0,  # Age in Gyr
        'mass': 1e10,  # Solar masses
        'sfh': 0,  # SSP
    }

    # Parameters for DL2007CigaleSSPBasis (non-parametric SFH)
    cigale_base_params = {
        'zred': 0.1,
        'logzsol': 0.0,
        'dust2': 0.3,
        'dust_type': 0,
        'dust_index': -0.7,
        'agebins': np.array([[0, 8], [8, 10.0]]),  # log(yr)
        'mass': np.array([1e9, 1e10]),  # Solar masses
    }

    print(f"\nStellar parameters:")
    print(f"  zred = {fsps_base_params['zred']}")
    print(f"  logzsol = {fsps_base_params['logzsol']}")
    print(f"  dust2 = {fsps_base_params['dust2']}")
    print(f"  FSPS: tage = {fsps_base_params['tage']} Gyr, mass = {fsps_base_params['mass']:.2e} Msun")
    print(f"  CIGALE: agebins-based SFH, total mass = {cigale_base_params['mass'].sum():.2e} Msun")

    # --- FSPS DL2007 ---
    print("\n--- Generating spectrum with FSPS DL2007 ---")
    try:
        sps_fsps = CSPSpecBasis(zcontinuous=1)

        # Set up parameters for FSPS
        fsps_params = fsps_base_params.copy()
        fsps_params['add_dust_emission'] = True
        fsps_params['duste_qpah'] = qpah
        fsps_params['duste_umin'] = umin
        fsps_params['duste_gamma'] = gamma

        wave_fsps, spec_fsps, mfrac_fsps = sps_fsps.get_galaxy_spectrum(**fsps_params)
        print(f"  Success! Wavelength range: {wave_fsps.min():.0f} - {wave_fsps.max():.0f} Angstroms")
        print(f"  Stellar mass fraction: {mfrac_fsps:.4f}")

    except Exception as e:
        print(f"  Error: {e}")
        wave_fsps, spec_fsps = None, None

    # --- CIGALE DL2007 ---
    print("\n--- Generating spectrum with CIGALE DL2007 ---")
    try:
        sps_cigale = DL2007CigaleSSPBasis(zcontinuous=1)

        # Set up parameters for CIGALE
        cigale_params = cigale_base_params.copy()
        cigale_params['dl2007_cigale_qpah'] = qpah
        cigale_params['dl2007_cigale_umin'] = umin
        cigale_params['dl2007_cigale_umax'] = umax
        cigale_params['dl2007_cigale_gamma'] = gamma

        wave_cigale, spec_cigale, mfrac_cigale = sps_cigale.get_galaxy_spectrum(**cigale_params)
        print(f"  Success! Wavelength range: {wave_cigale.min():.0f} - {wave_cigale.max():.0f} Angstroms")
        print(f"  Stellar mass fraction: {mfrac_cigale:.4f}")
        print(f"  L_absorbed: {sps_cigale.L_absorbed:.4e} L_sun/Msun")
        print(f"  <U>: {sps_cigale.umean:.2f}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        wave_cigale, spec_cigale = None, None

    # --- Compare ---
    if wave_fsps is not None and wave_cigale is not None:
        print("\n--- Comparison ---")

        # Interpolate CIGALE onto FSPS wavelength grid for comparison
        spec_cigale_interp = np.interp(wave_fsps, wave_cigale, spec_cigale)

        # Compare in different wavelength regions
        regions = [
            ('UV-Optical', 1000, 10000),
            ('Near-IR', 10000, 50000),
            ('Mid-IR', 50000, 300000),
            ('Far-IR', 300000, 5000000),
        ]

        print("\nRatio (CIGALE/FSPS) by wavelength region:")
        for name, wmin, wmax in regions:
            mask = (wave_fsps >= wmin) & (wave_fsps <= wmax)
            if mask.sum() > 0:
                # Use median ratio to avoid outliers
                ratio = np.median(spec_cigale_interp[mask] / spec_fsps[mask])
                print(f"  {name} ({wmin/1e4:.1f}-{wmax/1e4:.0f} μm): {ratio:.4f}")

        # Total luminosity comparison
        C_AA = 2.998e18
        nu = C_AA / wave_fsps
        L_fsps = -np.trapz(spec_fsps, nu)
        L_cigale = -np.trapz(spec_cigale_interp, nu)
        print(f"\nTotal luminosity (L_sun/Msun):")
        print(f"  FSPS:   {L_fsps:.4e}")
        print(f"  CIGALE: {L_cigale:.4e}")
        print(f"  Ratio:  {L_cigale/L_fsps:.4f}")

        # Find peak of dust emission (IR)
        ir_mask = wave_fsps > 50000  # > 5 microns
        if ir_mask.sum() > 0:
            peak_fsps = wave_fsps[ir_mask][np.argmax(spec_fsps[ir_mask])]
            peak_cigale = wave_fsps[ir_mask][np.argmax(spec_cigale_interp[ir_mask])]
            print(f"\nIR peak wavelength:")
            print(f"  FSPS:   {peak_fsps/1e4:.1f} μm")
            print(f"  CIGALE: {peak_cigale/1e4:.1f} μm")

        if plot:
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Full spectrum
                ax = axes[0, 0]
                ax.loglog(wave_fsps/1e4, spec_fsps, 'b-', label='FSPS DL2007', alpha=0.7)
                ax.loglog(wave_fsps/1e4, spec_cigale_interp, 'r--', label='CIGALE DL2007', alpha=0.7)
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel('L_ν (L_sun/Hz/Msun)')
                ax.set_title('Full Spectrum Comparison')
                ax.legend()
                ax.set_xlim(0.1, 1000)
                ax.grid(True, alpha=0.3)

                # IR region zoom
                ax = axes[0, 1]
                ir_mask = (wave_fsps > 10000) & (wave_fsps < 5e6)
                ax.loglog(wave_fsps[ir_mask]/1e4, spec_fsps[ir_mask], 'b-', label='FSPS DL2007', alpha=0.7)
                ax.loglog(wave_fsps[ir_mask]/1e4, spec_cigale_interp[ir_mask], 'r--', label='CIGALE DL2007', alpha=0.7)
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel('L_ν (L_sun/Hz/Msun)')
                ax.set_title('IR Region (1-500 μm)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Ratio
                ax = axes[1, 0]
                valid = (spec_fsps > 0) & (spec_cigale_interp > 0)
                ratio = np.ones_like(spec_fsps)
                ratio[valid] = spec_cigale_interp[valid] / spec_fsps[valid]
                ax.semilogx(wave_fsps/1e4, ratio, 'k-', alpha=0.7)
                ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel('Ratio (CIGALE/FSPS)')
                ax.set_title('Spectrum Ratio')
                ax.set_xlim(0.1, 1000)
                ax.set_ylim(0, 3)
                ax.grid(True, alpha=0.3)

                # Difference
                ax = axes[1, 1]
                diff = spec_cigale_interp - spec_fsps
                ax.semilogx(wave_fsps/1e4, diff, 'k-', alpha=0.7)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel('Difference (CIGALE - FSPS)')
                ax.set_title('Spectrum Difference')
                ax.set_xlim(0.1, 1000)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('dl2007_comparison.png', dpi=150)
                print(f"\nPlot saved to: dl2007_comparison.png")
                plt.show()

            except ImportError:
                print("\nMatplotlib not available for plotting.")

    return wave_fsps, spec_fsps, wave_cigale, spec_cigale


def compare_parameter_variations():
    """
    Compare how both implementations respond to parameter changes.
    """
    print("\n" + "=" * 70)
    print("Parameter Sensitivity Comparison")
    print("=" * 70)

    if not HAS_PROSPECT:
        print("Skipping (requires FSPS)")
        return

    # Base parameters for CIGALE DL2007 (non-parametric SFH)
    base_params = {
        'zred': 0.1,
        'logzsol': 0.0,
        'dust2': 0.3,
        'dust_type': 0,
        'dust_index': -0.7,
        'agebins': np.array([[0.0, 8.0], [8.0, 10.0]]),
        'mass': np.array([1e9, 1e10]),
    }

    qpah_base, umin_base, gamma_base = 2.50, 1.0, 0.1

    # Test qpah variation
    print("\n--- qpah variation ---")
    qpah_values = [0.47, 1.77, 2.50, 3.90, 4.58]

    for qpah in qpah_values:
        try:
            # CIGALE
            sps = DL2007CigaleSSPBasis(zcontinuous=1)
            params = base_params.copy()
            params['dl2007_cigale_qpah'] = qpah
            params['dl2007_cigale_umin'] = umin_base
            params['dl2007_cigale_umax'] = 1e6
            params['dl2007_cigale_gamma'] = gamma_base
            wave, spec, _ = sps.get_galaxy_spectrum(**params)

            # Find IR peak
            ir_mask = wave > 50000
            peak = wave[ir_mask][np.argmax(spec[ir_mask])] / 1e4
            print(f"  qpah={qpah:.2f}: IR peak at {peak:.1f} μm")

        except Exception as e:
            print(f"  qpah={qpah:.2f}: Error - {e}")

    # Test umin variation
    print("\n--- umin variation ---")
    umin_values = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0]

    for umin in umin_values:
        try:
            sps = DL2007CigaleSSPBasis(zcontinuous=1)
            params = base_params.copy()
            params['dl2007_cigale_qpah'] = qpah_base
            params['dl2007_cigale_umin'] = umin
            params['dl2007_cigale_umax'] = 1e6
            params['dl2007_cigale_gamma'] = gamma_base
            wave, spec, _ = sps.get_galaxy_spectrum(**params)

            ir_mask = wave > 50000
            peak = wave[ir_mask][np.argmax(spec[ir_mask])] / 1e4
            print(f"  umin={umin:.1f}: IR peak at {peak:.1f} μm, <U>={sps.umean:.2f}")

        except Exception as e:
            print(f"  umin={umin:.1f}: Error - {e}")

    # Test gamma variation
    print("\n--- gamma variation ---")
    gamma_values = [0.01, 0.05, 0.1, 0.3, 0.5]

    for gamma in gamma_values:
        try:
            sps = DL2007CigaleSSPBasis(zcontinuous=1)
            params = base_params.copy()
            params['dl2007_cigale_qpah'] = qpah_base
            params['dl2007_cigale_umin'] = umin_base
            params['dl2007_cigale_umax'] = 1e6
            params['dl2007_cigale_gamma'] = gamma
            wave, spec, _ = sps.get_galaxy_spectrum(**params)

            ir_mask = wave > 50000
            peak = wave[ir_mask][np.argmax(spec[ir_mask])] / 1e4
            print(f"  gamma={gamma:.2f}: IR peak at {peak:.1f} μm, <U>={sps.umean:.2f}")

        except Exception as e:
            print(f"  gamma={gamma:.2f}: Error - {e}")


def main():
    parser = argparse.ArgumentParser(description='Compare DL2007 implementations')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--templates-only', action='store_true',
                        help='Only test template loading (no FSPS required)')
    args = parser.parse_args()

    print("DL2007 Implementation Comparison")
    print("FSPS built-in vs CIGALE-ported templates")
    print("=" * 70)

    # Always test template loading
    compare_templates_only()

    if not args.templates_only and HAS_PROSPECT:
        # Full comparison with SPS
        compare_full_spectra(plot=args.plot)
        compare_parameter_variations()
    elif not HAS_PROSPECT:
        print("\n" + "=" * 70)
        print("Full spectrum comparison skipped (FSPS/sedpy not available)")
        print("Run with proper environment to test full SPS integration")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
