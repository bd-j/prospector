# Dust Emission Template Visualization Scripts

These scripts allow you to visualize and compare the dust emission templates implemented in Prospector.

## Scripts

### 1. `visualize_dl2007_comparison.py`

Compares FSPS's native DL2007 dust emission with the CIGALE-based DL2007 implementation.

```bash
conda run -n qso_fitting python scripts/visualize_dl2007_comparison.py
```

**Shows:**
- Raw CIGALE template data (minmin and minmax components)
- Side-by-side comparison of FSPS vs CIGALE for varying qpah, umin, and gamma
- Templates are normalized to the same integral for shape comparison

### 2. `visualize_dust_templates.py`

Visualizes all CIGALE-based dust emission templates.

```bash
# Show all models
conda run -n qso_fitting python scripts/visualize_dust_templates.py all

# Show specific model
conda run -n qso_fitting python scripts/visualize_dust_templates.py dl2014
conda run -n qso_fitting python scripts/visualize_dust_templates.py dale2014
conda run -n qso_fitting python scripts/visualize_dust_templates.py themis
conda run -n qso_fitting python scripts/visualize_dust_templates.py casey2012
```

**Models:**
- **DL2014**: Updated Draine & Li 2007 with variable alpha parameter
- **Dale2014**: Single-parameter (alpha) dust emission model
- **Themis**: Jones et al. 2017 model using HAC grains instead of PAHs
- **Casey2012**: Analytic modified blackbody + MIR power law

## Key Files to Review

The template loaders that may need debugging:

| File | Description |
|------|-------------|
| `prospect/sources/dl2007.py` | CIGALE DL2007 template loader |
| `prospect/sources/dl2014.py` | DL2014 template loader |
| `prospect/sources/dale2014.py` | Dale2014 template loader |
| `prospect/sources/themis.py` | Themis template loader |
| `prospect/sources/dl2007_basis.py` | DL2007 SSP basis class |

The build scripts that create the template files:

| File | Output |
|------|--------|
| `scripts/build_dl2007_templates.py` | `dust_data/dl2007/templates.npz` |
| `scripts/build_dl2014_templates.py` | `dust_data/dl2014/templates.npz` |
| `scripts/build_dale2014_templates.py` | `dust_data/dale2014/templates.npz` |
| `scripts/build_themis_templates.py` | `dust_data/themis/templates.npz` |

## Code Review

To view code changes on GitHub:
```bash
git diff main..fix-speccal-ordering -- prospect/sources/
```

Or push the branch and view on GitHub:
```bash
git push -u origin fix-speccal-ordering
# Then view at: https://github.com/[repo]/compare/main...fix-speccal-ordering
```

## Known Issues to Investigate

1. **Unit conversion in template loaders**: The raw CIGALE templates are in W/nm (F_lambda). The loaders convert to L_sun/Hz (F_nu) using:
   ```python
   F_nu = F_lambda * lambda^2 / c
   ```
   Then normalize so the integral equals 1 L_sun.

2. **DL2007CigaleSSPBasis**: Uses FastStepBasis (non-parametric SFH). May have issues with how absorbed luminosity is calculated and applied.

3. **Template grid spacing**: The CIGALE templates use discrete parameter grids that may differ from FSPS's continuous interpolation.
