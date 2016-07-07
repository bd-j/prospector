Requirements
============

|Codename| has not been tested with Python3 yet, please use 2.7.x

You will also need:

-  `numpy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_

-  `emcee <http://dan.iel.fm/emcee/current/>`_ (Please cite this package in any publications)

-  `sedpy <https://github.com/bd-j/sedpy>`_ (for nonparametric SFHs and SSPs)

For more portable output files, or for modeling stars, you will need:

- `HDF5 <https://www.hdfgroup.org/HDF5/>`_ and `h5py <http://www.h5py.org>`_
  (If you have Enthought or Anaconda one or both of these may already be installed,
  or you can get HDF5 from homebrew or macports)

For modeling galaxies you will need:

-  `FSPS <https://github.com/cconroy20/fsps>`_ and
   `python-FSPS <https://github.com/dfm/python-FSPS>`_

You may also wish to have `AstroPy <https://astropy.readthedocs.org/en/stable/>`_
for FITS file processing and cosmological calculations.

For parallel processing you will need:

-  MPI (e.g. openMPI or mvapich2, available from homebrew, macports, or Anaconda)  and
   `mpi4py <http://pythonhosted.org/mpi4py/>`_

Installation
==========

Then just git clone the repo and make sure it is somewhere in your
python path. |Codename| is pure python.

.. |Codename| replace:: Prospector
