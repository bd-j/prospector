Prospector
=====
Conduct principled inference of stellar population properties from photometric
and/or spectroscopic data.  Prospector allows you to:

* Infer high-dimensional stellar population properties using parameteric or nonparametric SFHs
  (with nested or ensemble MCMC sampling)

* Use spectra and/or photometry to constrain the linear combination of stellar population
  components that are present in a galaxy (i.e. non-parametric SFHs).

* Combine photometric and spectroscopic data rigorously using a flexible
  spectroscopic calibration model.

* Forward model many aspects of spectroscopic data analysis and
  calibration, including spectrophotometric calibration, sky emission (coming soon),
  and wavelength solution, thus properly incorporating uncertainties
  in these components in the final  parameter uncertainties.

Read the documentation [here](http://prospect.readthedocs.io/en/latest/).

Example
-------
Inference with mock broadband data, showing the change in posteriors as the
number of photometric bands is increased.
![Demonstration of posteriro inference with increasing number of photometric bands](doc/images/animation.gif)


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
Other files in the [doc/](doc/) directory explain the usage of the code,
and you can read the documentation [here](http://prospect.readthedocs.io/en/latest/).

See also the [tutorial](demo/tutorial.rst), the [demo notebook](demo/NestedDemo.ipynb),
or the [deconstructed demo](demo/DeconstructedDemo.ipynb)
for fitting photometric data with composite stellar populations.




Citation
------
If you use this code, please reference the doi below,
and make sure to cite the dependencies as listed in [installation](doc/installation.rst)
[![DOI](https://zenodo.org/badge/10490445.svg)](https://zenodo.org/badge/latestdoi/10490445)

You should also cite:
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
