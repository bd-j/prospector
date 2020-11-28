Advanced Usage
==============

Spectral line marginalization
------------------------------
Accurately predicting nebular line fluxes can be challenging. The
:py:class:`prospect.models.sedmodel.SpecModel` class can be used to determine
the maximum-likelihood line amplitudes for each predicition, and to compute a
likelihood penalty for marginalizing over all possible line amplitudes. It is
even possible to incorporate priors based on the FSPS nebular line model.  Note
that line velocity offsets and widths must still be explicitly fit for (or
specified) as model parameters.


Noise Modeling
-------------------
This is handled by specifiying rules for constructing a covariance matrix, and
supplying a ``build_noise()`` method in the parameter file.


Mock data
---------------
Really this should not be advanced. Everyone should do mock data tests. So we
are trying to make it easy. See demo/demo_mock_params.py for a suggestion,
especially the ``load_obs()`` function.


MPI
------
When sampling with emcee it is possible to parallelize the computations over many processors.
Install some kind of MPI on your system (openMPI, mpich2, mvapich2),
make sure mpi4py is also installed against this MPI installation,
and use the syntax
``mpirun -np <N> python <mpi_param_file>``

This causes likelihood evaluations for different walkers to be made in parallel.
For optimal results, the number of emcee walkers should be :math:`2*N*(N_p-1)`,
where N is an integer and :math:`N_p` is the number of available processors.

Note that specific MPI implementations may have different mpirun commands, or
may require that python-mpi be called instead of just python.  We have included
a small script (``demo/mpi_hello_world.py``) to test your MPI installation
using same general pattern as in Prospector.  Run this with
``mpirun -np <N> python mpi_hello_world.py``.


User defined sources
--------------------------
It's possible to replace the default ``source`` objects, which are wrappers on
python-FSPS, with your own sources. For example, stars instead of stellar
populations, or quasar spectra or planets.

The only requirement on your ``sources`` object class is that it should have a
``get_galaxy_spectrum(outwave=[], filters=[], **params)``
method that converts a dictionary of parameters, a list of filters, and a
wavelength grid into a model SED and spectrum, and returns the spectrum, the
photometry, and any ancillary info. You will have to write that.

See any of the ``sources`` classes for the appropriate ``get_galaxy_spectrum`` API.


Multiple Spectra
----------------------
We are working on this.


Outlier modeling
----------------


.. |Codename| replace:: Prospector
