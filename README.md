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

Read the documentation [here](http://prospect.readthedocs.io/en/latest/).

Installation
------------

To install to a conda environment with dependencies, see `conda_install.sh`.

To install just Prospector:
```
python -m pip install astro-prospector
```

To install the latest development version:
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

See [installation](doc/installation.rst) for requirements.
Other files in the [doc/](doc/) directory explain the usage of the code,
and you can read the documentation [here](http://prospect.readthedocs.io/en/latest/).

See also the [tutorial](demo/tutorial.rst)
or the [interactive demo](demo/InteractiveDemo.ipynb)
for fitting photometric data with composite stellar populations.

Example
-------

Inference with mock broadband data, showing the change in posteriors as the
number of photometric bands is increased.
![Demonstration of posterior inference with increasing number of photometric bands](doc/images/animation.gif)

Citation
------

If you use this code, please reference
```
@MISC{2019ascl.soft05025J,
       author = {{Johnson}, Benjamin D. and {Leja}, Joel L. and {Conroy}, Charlie and
         {Speagle}, Joshua S.},
        title = "{Prospector: Stellar population inference from spectra and SEDs}",
     keywords = {Software},
         year = 2019,
        month = may,
          eid = {ascl:1905.025},
        pages = {ascl:1905.025},
archivePrefix = {ascl},
       eprint = {1905.025},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019ascl.soft05025J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

and make sure to cite the dependencies as listed in [installation](doc/installation.rst)

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
