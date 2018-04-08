Requirements
============

|Codename| is developed on Python2.7 but should work with Python3

You will also need:

-  `numpy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_

-  `emcee <http://dan.iel.fm/emcee/current/>`_ and/or `dynesty <https://dynesty.readthedocs.io/en/latest/>`_ for inference (Please cite these packages in any publications)

-  `sedpy <https://github.com/bd-j/sedpy>`_ (for filter projections)

For portable output files, or for modeling stars, you will need:

- `HDF5 <https://www.hdfgroup.org/HDF5/>`_ and `h5py <http://www.h5py.org>`_
  (If you have Enthought or Anaconda one or both of these may already be installed,
  or you can get HDF5 from homebrew or macports)

For modeling galaxies you will need:

-  `FSPS <https://github.com/cconroy20/fsps>`_ and
   `python-FSPS <https://github.com/dfm/python-FSPS>`_ (Please cite these packages in any publications)

You may also wish to have `AstroPy <https://astropy.readthedocs.org/en/stable/>`_
for FITS file processing and cosmological calculations, please cite this package in any publications.

For parallel processing with emcee you will need:

-  MPI (e.g. openMPI or mvapich2, available from homebrew, macports, or Anaconda)  and
   `mpi4py <http://pythonhosted.org/mpi4py/>`_

Installation
==========

|Codename| is pure python.

.. code-block:: shell

		cd <install_dir>
		git clone https://github.com/bd-j/prospector
		cd prospector
		python setup.py install

Then in Python

.. code-block:: python

		import prospect

.. |Codename| replace:: Prospector
