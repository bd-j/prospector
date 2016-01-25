BSFH
=====
Conduct principled inference of stellar population properties from photometric
and/or spectroscopic data.  BSFH allows you to:

* Combine photometric and spectroscopic data rigorously using a flexible
  spectroscopic calibration model.

* Infer high-dimensional stellar population properties using parameteric SFHs
  (with ensemble MCMC sampling)

* Use spectra to constrain the linear combination of stellar population
  components that are present in a galaxy (e.g. non-parametric SFHs) using
  Hybrid Monte Carlo.

* Fit individual stellar spectra using large (in both n and d) interpolated
  grids.

* Forward model many aspects of spectroscopic data analysis and
  calibration, including sky emission, spectrophotometric calibration,
  and wavelength solutions, thus properly incorporating uncertainties
  in these components in the final  parameter uncertainties.

Installation
------
```
cd <install_dir>
git clone https://github.com/bd-j/bsfh
```

Then in C-Shell:
```
setenv $PYTHONPATH:+<install_dir>/bsfh
```
or in bash:
```
export $PYTHONPATH=$PYTHONPATH:<install_dir>/bsfh
```

See [installation](doc/installation.rst) for requirements.
Other files in the [doc/](doc/) directory explain the usage of the code.

See also the [tutorial](demo/tutorial.rst) for fitting photometric data with composite stellar populations.
