Installation
============

|Codename| itself is is pure python.  To install a released version of just
|Codename|, use ``pip``

.. code-block:: shell

		python -m pip install astro-prospector

Then in Python

.. code-block:: python

        import prospect
        print(prospect.__version__)

However, several other packages are required for the code to model and fit SEDs
(see below.)

Development Version
-------------------

To install the development version of |Codename| and its dependencies to a conda
environment, use the following procedure:

.. code-block:: shell

        # change this if you want to install elsewhere;
        # or, copy and run this script in the desired location
        CODEDIR=$PWD
        cd $CODEDIR

        # Clone FSPS to get data files
        git clone git@github.com:cconroy20/fsps
        export SPS_HOME="$PWD/fsps"

        # Create and activate environment (here named 'prospector')
        git clone git@github.com:bd-j/prospector.git
        cd prospector
        conda env create -f environment.yml -n prospector
        conda activate prospector
        # Install latest development version of prospector
        python -m pip uninstall astro-prospector
        python -m pip install .

        echo "Add 'export SPS_HOME=$SPS_HOME' to your .bashrc"

        # To use prospector activate the conda environment
        conda activate prospector


Requirements
------------

|Codename| works with ``python>=3.10``, and requires `numpy
<http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_

You will also need:


- `astropy <https://astropy.readthedocs.org/en/stable/>`_

-  `emcee <https://emcee.readthedocs.io/en/stable/>`_ and/or `dynesty <https://dynesty.readthedocs.io/en/latest/>`_
   for inference (Please cite these packages in any publications)

-  `sedpy <https://github.com/bd-j/sedpy>`_ (for filter projections)

- `HDF5 <https://www.hdfgroup.org/HDF5/>`_ and `h5py <http://www.h5py.org>`_
  (If you have Enthought or Anaconda one or both of these may already be
  installed, or you can get HDF5 from homebrew or macports and h5py via pip)

For modeling galaxies you will need:

-  `FSPS <https://github.com/cconroy20/fsps>`_ and
   `python-FSPS <https://github.com/dfm/python-FSPS>`_ (Please cite these packages in any publications)


For parallel processing with emcee or dynesty (optional) you will need:

-  MPI (e.g. openMPI or mvapich2, available from homebrew, macports, or Anaconda)  and
   `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_




.. |Codename| replace:: Prospector
