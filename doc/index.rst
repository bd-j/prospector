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

*Copyright 2014-2018 Benjamin D. Johnson and contributors.*

This code is available under the `MIT License
<https://raw.github.com/bdj/prospector/master/LICENSE.rst>`_.
