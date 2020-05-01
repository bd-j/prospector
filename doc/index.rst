Prospector
======================================
**Prospector** is a package to conduct principled inference of stellar population
properties from photometric and/or spectroscopic data using flexible models.
Prospector allows you to:

* Infer high-dimensional stellar population properties, including nebular
  emission, from rest UV through Far-IR data (with nested or ensemble MCMC sampling.)

* Combine photometric and spectroscopic data rigorously using a flexible
  spectroscopic calibration model.

* Use spectra and/or photometry to constrain the linear combination of stellar
  population components that are present in a galaxy (e.g. non-parametric
  SFHs).

* Forward model many aspects of spectroscopic data analysis and calibration,
  including spectral resolution, spectrophotometric calibration, sky emission
  (coming soon), and wavelength solution, thus properly incorporating
  uncertainties in these components in the final parameter uncertainties.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   dataformat
   models
   output

.. toctree::
   :maxdepth: 1
   :caption: Demos & Tutorials

   demo
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/models_api
   api/sources_api
   api/fitting_api
   api/io_api
   api/utils_api

License and Attribution
------------------------------

*Copyright 2014-2020 Benjamin D. Johnson and contributors.*

This code is available under the `MIT License
<https://raw.github.com/bdj/prospector/master/LICENSE.rst>`_.

If you use this code, please reference

.. code-block:: none

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

