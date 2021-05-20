<img src="doc/images/logo_name.png" height=75/>  <!-- . -->
==========

Conduct principled inference of stellar population properties from photometric
and/or spectroscopic data.  Prospector allows you to:

* Infer high-dimensional stellar population properties using parameteric or nonparametric SFHs
  (with nested or ensemble MCMC sampling)

* Combine photometric and spectroscopic data from the UV to Far-IR rigorously using a flexible
  spectroscopic calibration model.

* Forward model many aspects of spectroscopic data analysis and
  calibration, including spectrophotometric calibration and wavelength solution,
  thus properly incorporating uncertainties in these components in the final parameter uncertainties.

Read the [documentation](http://prospect.readthedocs.io/en/latest/) and the
code [paper](https://ui.adsabs.harvard.edu/abs/2020arXiv201201426J/abstract).

Installation
------------

See [installation](doc/installation.rst) for requirements and dependencies.
The [documentation](http://prospect.readthedocs.io/en/latest/) includes a tutorial and demos.

To install to a conda environment with dependencies, see `conda_install.sh`.
To install just Prospector (stable release):
```
python -m pip install astro-prospector
```

To install the latest development version:
```
cd <install_dir>
git clone https://github.com/bd-j/prospector
cd prospector
python -m pip install .
```

Then, in Python
```python
import prospect
```


Citation
------

If you use this code, please reference [this paper](https://ui.adsabs.harvard.edu/abs/2020arXiv201201426J/abstract):
```
@ARTICLE{2020arXiv201201426J,
       author = {{Johnson}, Benjamin D. and {Leja}, Joel and {Conroy}, Charlie and {Speagle}, Joshua S.},
        title = "{Stellar Population Inference with Prospector}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2020,
        month = dec,
          eid = {arXiv:2012.01426},
        pages = {arXiv:2012.01426},
archivePrefix = {arXiv},
       eprint = {2012.01426},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201201426J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

and make sure to cite the dependencies as listed in [installation](doc/installation.rst)

Example
-------

Inference with mock broadband data, showing the change in posteriors as the
number of photometric bands is increased.
![Demonstration of posterior inference with increasing number of photometric bands](doc/images/animation.gif)
