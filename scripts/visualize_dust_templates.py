#!/usr/bin/env python
"""
Visualize all CIGALE-based dust emission templates.

This script plots dust emission templates for:
- DL2014 (Draine & Li 2007 updated with variable alpha)
- Dale2014 (Dale et al. 2014 single-parameter model)
- Themis (Jones et al. 2017)
- Casey2012 (Modified blackbody)

Usage:
    python visualize_dust_templates.py [model]

    Where [model] is one of: dl2014, dale2014, themis, casey2012, all
    Default: all

Requires: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)


def plot_dl2014():
    """Visualize DL2014 templates."""
    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/dl2014/templates.npz')
    if not os.path.exists(template_path):
        print(f"DL2014 templates not found at {template_path}")
        return

    exec(open(os.path.join(project_dir, 'prospect/sources/dl2014.py')).read(), globals())
    DL2014Templates._instance = None
    dl2014 = DL2014Templates(template_path)

    print("\nDL2014 Template Info:")
    print(f"  Wavelength: {dl2014.wavelength.min():.0f} - {dl2014.wavelength.max():.0f} Angstrom")
    print(f"  qpah values ({len(dl2014.qpah_values)}): {dl2014.qpah_values}")
    print(f"  umin values ({len(dl2014.umin_values)}): {dl2014.umin_values.min():.2f} - {dl2014.umin_values.max():.1f}")
    print(f"  alpha values ({len(dl2014.alpha_values)}): {dl2014.alpha_values}")

    wave_um = dl2014.wavelength / 1e4
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DL2014 Dust Emission Templates', fontsize=14)

    # Vary qpah
    ax = axes[0, 0]
    for qpah in [0.47, 1.77, 3.19, 5.26, 7.32]:
        _, spec, _ = dl2014.get_template(qpah, umin=1.0, alpha=2.0, gamma=0.1)
        ax.loglog(wave_um, spec * wave_um, label=f'qpah={qpah:.2f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary qpah (umin=1, α=2, γ=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Vary umin
    ax = axes[0, 1]
    for umin in [0.1, 0.5, 1.0, 5.0, 25.0]:
        _, spec, _ = dl2014.get_template(qpah=2.5, umin=umin, alpha=2.0, gamma=0.1)
        ax.loglog(wave_um, spec * wave_um, label=f'umin={umin:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary umin (qpah=2.5, α=2, γ=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Vary alpha
    ax = axes[1, 0]
    for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
        _, spec, _ = dl2014.get_template(qpah=2.5, umin=1.0, alpha=alpha, gamma=0.1)
        ax.loglog(wave_um, spec * wave_um, label=f'α={alpha:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary alpha (qpah=2.5, umin=1, γ=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Vary gamma
    ax = axes[1, 1]
    for gamma in [0.01, 0.05, 0.1, 0.3, 0.5]:
        _, spec, _ = dl2014.get_template(qpah=2.5, umin=1.0, alpha=2.0, gamma=gamma)
        ax.loglog(wave_um, spec * wave_um, label=f'γ={gamma:.2f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary gamma (qpah=2.5, umin=1, α=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_dale2014():
    """Visualize Dale2014 templates."""
    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/dale2014/templates.npz')
    if not os.path.exists(template_path):
        print(f"Dale2014 templates not found at {template_path}")
        return

    exec(open(os.path.join(project_dir, 'prospect/sources/dale2014.py')).read(), globals())
    Dale2014Templates._instance = None
    dale2014 = Dale2014Templates(template_path)

    print("\nDale2014 Template Info:")
    print(f"  Wavelength: {dale2014.wavelength.min():.0f} - {dale2014.wavelength.max():.0f} Angstrom")
    print(f"  alpha values ({len(dale2014.alpha_values)}): {dale2014.alpha_values.min():.3f} - {dale2014.alpha_values.max():.1f}")

    wave_um = dale2014.wavelength / 1e4
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Dale2014 Dust Emission Templates', fontsize=14)

    # Plot for various alpha values
    alpha_subset = [0.0625, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_subset)))

    for alpha, color in zip(alpha_subset, colors):
        _, spec = dale2014.get_template(alpha)
        ax.loglog(wave_um, spec * wave_um, label=f'α={alpha:.2f}', color=color)

    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized to 1 L_sun)')
    ax.set_title('Dale2014: Single-parameter dust model\n(Lower α = warmer dust, Higher α = cooler dust)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.show()


def plot_themis():
    """Visualize Themis templates."""
    template_path = os.path.join(project_dir, 'prospect/sources/dust_data/themis/templates.npz')
    if not os.path.exists(template_path):
        print(f"Themis templates not found at {template_path}")
        return

    exec(open(os.path.join(project_dir, 'prospect/sources/themis.py')).read(), globals())
    ThemisTemplates._instance = None
    themis = ThemisTemplates(template_path)

    print("\nThemis Template Info:")
    print(f"  Wavelength: {themis.wavelength.min():.0f} - {themis.wavelength.max():.0f} Angstrom")
    print(f"  qhac values ({len(themis.qhac_values)}): {themis.qhac_values}")
    print(f"  umin values ({len(themis.umin_values)}): {themis.umin_values.min():.2f} - {themis.umin_values.max():.1f}")
    print(f"  alpha values ({len(themis.alpha_values)}): {themis.alpha_values}")

    wave_um = themis.wavelength / 1e4
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Themis Dust Emission Templates\n(Uses HAC grains instead of PAHs)', fontsize=14)

    # Vary qhac
    ax = axes[0, 0]
    for qhac in [0.02, 0.10, 0.17, 0.28, 0.40]:
        _, spec, _ = themis.get_template(qhac, umin=1.0, alpha=2.0, gamma=0.1)
        ax.loglog(wave_um, spec * wave_um, label=f'qhac={qhac:.2f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary qhac (umin=1, α=2, γ=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Vary umin
    ax = axes[0, 1]
    for umin in [0.1, 0.5, 1.0, 5.0, 25.0]:
        _, spec, _ = themis.get_template(qhac=0.17, umin=umin, alpha=2.0, gamma=0.1)
        ax.loglog(wave_um, spec * wave_um, label=f'umin={umin:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary umin (qhac=0.17, α=2, γ=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Vary alpha
    ax = axes[1, 0]
    for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
        _, spec, _ = themis.get_template(qhac=0.17, umin=1.0, alpha=alpha, gamma=0.1)
        ax.loglog(wave_um, spec * wave_um, label=f'α={alpha:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary alpha (qhac=0.17, umin=1, γ=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    # Vary gamma
    ax = axes[1, 1]
    for gamma in [0.01, 0.05, 0.1, 0.3, 0.5]:
        _, spec, _ = themis.get_template(qhac=0.17, umin=1.0, alpha=2.0, gamma=gamma)
        ax.loglog(wave_um, spec * wave_um, label=f'γ={gamma:.2f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary gamma (qhac=0.17, umin=1, α=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_casey2012():
    """Visualize Casey2012 modified blackbody model."""
    # Casey2012 is analytic, not template-based
    # Check if the module exists
    casey_path = os.path.join(project_dir, 'prospect/sources/casey2012.py')
    if not os.path.exists(casey_path):
        print(f"Casey2012 module not found at {casey_path}")
        return

    exec(open(casey_path).read(), globals())

    print("\nCasey2012 Model Info:")
    print("  Analytic modified blackbody + MIR power law")
    print("  Parameters: temperature, alpha, beta")

    # Create Casey2012 model instance
    model = Casey2012Model()

    # Create wavelength grid (in Angstroms)
    wave = np.logspace(np.log10(1e4), np.log10(1e8), 500)  # 1 um to 10 mm
    wave_um = wave / 1e4

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Casey2012 Modified Blackbody Dust Emission', fontsize=14)

    # Vary temperature
    ax = axes[0]
    for temp in [20, 30, 40, 50, 60]:
        _, spec = model.get_spectrum(temperature=temp, alpha=2.0, beta=1.5, target_wave=wave)
        ax.loglog(wave_um, spec * wave_um, label=f'T={temp} K')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary temperature (α=2, β=1.5)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9)

    # Vary alpha (MIR power-law slope)
    ax = axes[1]
    for alpha in [1.5, 2.0, 2.5, 3.0]:
        _, spec = model.get_spectrum(temperature=35, alpha=alpha, beta=1.5, target_wave=wave)
        ax.loglog(wave_um, spec * wave_um, label=f'α={alpha:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary α (T=35K, β=1.5)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9)

    # Vary beta (emissivity index)
    ax = axes[2]
    for beta in [1.0, 1.5, 2.0, 2.5]:
        _, spec = model.get_spectrum(temperature=35, alpha=2.0, beta=beta, target_wave=wave)
        ax.loglog(wave_um, spec * wave_um, label=f'β={beta:.1f}')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('λ F_λ (normalized)')
    ax.set_title('Vary β (T=35K, α=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    model = 'all'
    if len(sys.argv) > 1:
        model = sys.argv[1].lower()

    print("=" * 60)
    print("Dust Emission Template Visualization")
    print("=" * 60)

    if model in ['dl2014', 'all']:
        print("\n--- DL2014 ---")
        plot_dl2014()

    if model in ['dale2014', 'all']:
        print("\n--- Dale2014 ---")
        plot_dale2014()

    if model in ['themis', 'all']:
        print("\n--- Themis ---")
        plot_themis()

    if model in ['casey2012', 'all']:
        print("\n--- Casey2012 ---")
        try:
            plot_casey2012()
        except Exception as e:
            print(f"Could not plot Casey2012: {e}")

    if model not in ['dl2014', 'dale2014', 'themis', 'casey2012', 'all']:
        print(f"Unknown model: {model}")
        print("Valid options: dl2014, dale2014, themis, casey2012, all")


if __name__ == "__main__":
    main()
