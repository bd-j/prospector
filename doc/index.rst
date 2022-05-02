Prospector
======================================
**Prospector** is a package to conduct principled inference of stellar population
properties from photometric and/or spectroscopic data using flexible models.
Prospector allows you to:

* Infer high-dimensional stellar population properties, including nebular
  emission, from rest UV through Far-IR data (with nested or ensemble MCMC
  sampling.)

* Combine photometric and spectroscopic data rigorously using a flexible
  spectroscopic calibration model and forward modeling many
  aspects of spectroscopic data analysis.

* Use spectra and/or photometry to constrain highly flexible star formation
  history treatments.

.. image:: https://img.shields.io/badge/GitHub-bdj%2Fprospector-blue.svg
    :target: https://github.com/bd-j/prospector
.. image:: https://img.shields.io/badge/arXiv-2012.01426-b31b1b.svg
    :target: https://arxiv.org/abs/2012.01426
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/bd-j/prospector/blob/main/LICENSE


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   dataformat
   models
   sfhs
   noise
   output

.. toctree::
   :maxdepth: 1
   :caption: Demos & Tutorials

   quickstart
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

*Copyright 2014-2022 Benjamin D. Johnson and contributors.*

This code is available under the `MIT License
<https://raw.github.com/bdj/prospector/blob/main/LICENSE>`_.

If you use this code, please reference `this paper <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_:

.. code-block:: none

    @ARTICLE{2021ApJS..254...22J,
        author = {{Johnson}, Benjamin D. and {Leja}, Joel and {Conroy}, Charlie and {Speagle}, Joshua S.},
            title = "{Stellar Population Inference with Prospector}",
        journal = {\apjs},
        keywords = {Galaxy evolution, Spectral energy distribution, Astronomy data modeling, 594, 2129, 1859, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
            year = 2021,
            month = jun,
        volume = {254},
        number = {2},
            eid = {22},
            pages = {22},
            doi = {10.3847/1538-4365/abef67},
    archivePrefix = {arXiv},
        eprint = {2012.01426},
    primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Changelog
---------

.. include:: ../CHANGELOG.rst