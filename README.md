Prospector
=====
Conduct principled inference of stellar population properties from photometric
and/or spectroscopic data.  Prospector allows you to:

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
cd prospector
python setup.py install
```

Then in Python
```python
import prospect
```

Prospector is pure python.
See [installation](doc/installation.rst) for requirements.
Other files in the [doc/](doc/) directory explain the usage of the code.

See also the [tutorial](demo/tutorial.rst), the [interactive demo](demo/InteractiveDemo.ipynb),
or the [deconstructed demo](demo/DeconstructedDemo.ipynb) for fitting photometric data with composite stellar populations.


Citation
------
If you use this code, please reference the doi below, and make sure to cite the dependencies as listed in [installation](doc/installation.rst)
[![DOI](https://zenodo.org/badge/10490445.svg)](https://zenodo.org/badge/latestdoi/10490445)

You might also cite:
```
@article{2017ApJ...837..170L,
   author = {{Leja}, J. and {Johnson}, B.~D. and {Conroy}, C. and {van Dokkum}, P.~G. and {Byler}, N.},
   title = "{Deriving Physical Properties from Broadband Photometry with Prospector: Description of the Model and a Demonstration of its Accuracy Using 129 Galaxies in the Local Universe}",
   journal = {\apj},
   year = 2017,
   volume = 837,
   pages = {170},
   eprint = {1609.09073},
   doi = {10.3847/1538-4357/aa5ffe},
  adsurl = {http://adsabs.harvard.edu/abs/2017ApJ...837..170L},
}
```
