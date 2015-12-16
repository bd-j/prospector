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
  and wavelength solutions, thus proprly incorporating uncertainties
  in these components in the final  parameter uncertainties.

Requirements
-------
BSFH requires numpy, scipy, emcee, and h5py.  For galaxy modeling it requires
python-FSPS, and in some cases sedpy.  For stellar modeling it requires sedpy.
For parallel processing it requires mpi4py.  Visulaization requires matplotlib,
and optionally corner.py

Installation
------
```
cd <install_dir>
git clone https://github.com/bd-j/bsfh
setenv $PYTHONPATH:+<install_dir>/bsfh
```
