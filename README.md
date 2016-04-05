Prospector
=====
Conduct principled inference of stellar population properties from photometric
and/or spectroscopic data.  BSFH allows you to:

* Combine photometric and spectroscopic data rigorously using a flexible
  spectroscopic calibration model.

* Infer high-dimensional stellar population properties using parameteric SFHs
  (with ensemble MCMC sampling)

* Use spectra and/or photometry to constrain the linear combination of stellar population
  components that are present in a galaxy (e.g. non-parametric SFHs).

* Fit individual stellar spectra using large (in both n and d) interpolated
  grids, or polynomial spectral interpolators (coming soon).

* Forward model many aspects of spectroscopic data analysis and
  calibration, including spectrophotometric calibration, sky emission (coming soon),
  and wavelength solution (coming soon), thus properly incorporating uncertainties
  in these components in the final  parameter uncertainties.

Installation
------
```
cd <install_dir>
git clone https://github.com/bd-j/prospector
```

Then in C-Shell:
```
setenv $PYTHONPATH:+<install_dir>/prospector
```
or in bash:
```
export $PYTHONPATH=$PYTHONPATH:<install_dir>/prospector
```

Prospector is pure python.
See [installation](doc/installation.rst) for requirements.
Other files in the [doc/](doc/) directory explain the usage of the code.

See also the [tutorial](demo/tutorial.rst) for fitting photometric data with composite stellar populations.
