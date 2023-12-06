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
   nebular
   output
   ref

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

Changelog
---------

.. include:: ../CHANGELOG.rst