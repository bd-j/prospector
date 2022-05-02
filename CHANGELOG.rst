.. :changelog:

v1.1 (2022-02-20)
+++++++++++++++++

- Improved treatment of emission lines in ``SpecModel``, including ability to ignore
  selected lines entirely.
- New ``NoiseModelKDE`` and ``Kernel`` classes to accommodate non-Gaussian and
  correlated uncertainties, courtesy of @wpb-astro
- New flexible SFH parameterization courtesy @wrensuess
- Support for ``sedpy.observate.FilterSet`` objects and computing rest-frame
  absolute magnitudes.
- Documentation updates, including a dedicated SFH page and a quickstart.
- Several bugfixes including fixes to the "logm_sfh" parameter template, a fix
  for the nested sampling argument parsing, and bestfit spectrum saving.

v1.0 (2021-12-02)
+++++++++++++++++

Release to accompany submitted paper. Includes

- New plotting module
- Demonstrations of MPI usage with dynesty
- Numerous small bugfixes.

v0.4 (2021-07-08)
+++++++++++++++++

- New ``models.SpecModel`` class that handles much of the conversion from FSPS
  spectra to observed frame spectra (redshifting, smoothing, dimming,
  spectroscopic calibration, filter projections) internally instead of relying
  on source classes.
- The ``SpecModel`` class enables analytic marginalization of emission line
  amplitudes, with or without FSPS based priors.
- A new mixture model option in the likelihood to handle outlier points (for
  diagonal covariance matrices)
- A noise model kernel for photometric calibration offsets.
- Rename ``mean_model()`` to ``predict()`` (old method kept for backwards compatibility)
- Some fixes to priors and optimization
- Python3 compatibility improvements (now developed and tested with Python3)

v0.3 (2019-04-23)
+++++++++++++++++

- New UI, based on ``argparse`` command line options and a high level
  ``fit_model()` function that can use emcee, dynesty, or optimization algorithms
- New ``prospector_parse`` module that generates a default argument parser.
- Importable default probability function as ``fitting.lnprobfn()``
- Non-object prior methods removed
- Documentation and new notebook reflect UI changes
- ``model_setup`` methods are deprecated, better usage of warnings
