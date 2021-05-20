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
  including spectral resolution, spectrophotometric calibration and wavelength
  solution, thus properly incorporating uncertainties in these components in the
  final parameter uncertainties.


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
   faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/models_api
   api/sources_api
   api/fitting_api
   api/io_api
   api/plotting_api
   api/utils_api

License and Attribution
------------------------------

*Copyright 2014-2021 Benjamin D. Johnson and contributors.*

This code is available under the `MIT License
<https://raw.github.com/bdj/prospector/blob/main/LICENSE>`_.

If you use this code, please reference `this paper <https://ui.adsabs.harvard.edu/abs/2020arXiv201201426J/abstract>`_:

.. code-block:: none

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
