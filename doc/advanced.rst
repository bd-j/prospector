Advanced Usage
==============

Spectral line marginalization
------------------------------

Accurately predicting nebular line fluxes can be challenging.
The :py:class:`prospect.models.sedmodel.SpecModel` class can be used to
determine the maximum-likelihood line amplitudes for each predicition, and to
compute a likelihood penalty for marginalizing over all possible line
amplitudes. It is even possible to incorporate priors based on the FSPS nebular
line model.  Note that line velocity offsets and widths must still be explicitly
fit for (or specified) as model parameters.


Noise Modeling
-------------------

This is handled by specifiying rules for constructing a covariance matrix, and
supplying a ``load_noise()`` method in the parameter file.


Mock data
---------------

Really this should not be advanced.
Everyone should do mock data tests.
So we are trying to make it easy.
See demo/demo_mock_params.py for a suggestion, especially the ``load_obs()`` function.


MPI
------

This code is slow! Get moar processors.

When sampling with emcee it is possible to parallelize the computations over many processors.
Install some kind of MPI on your system (openMPI, mpich2, mvapich2),
make sure mpi4py is also installed against this MPI installation,
and use the syntax
``mpirun -np <N> python prospector.py –-param_file=<param_file>``

This causes likelihood evaluations for different walkers to be made in parallel.
For optimal results, the number of emcee walkers should be :math:`N*(np-1)`,
where N is an even integer and :math:`np` is the number of available processors.

Note that specific MPI implementations may have different mpirun commands, or
may require that python-mpi be called instead of just python.  We have included
a small script (``demo/mpi_hello_world.py``) to test your MPI installation
using same general pattern as in Prospector.  Run this with
``mpirun -np <N> python mpi_hello_world.py``.


User defined models
--------------------------

Problem: The pre-packaged models suck! You can do better.
Or, you have stars instead of stellar populations. Or spectra of the IGM or
planets or AGN or something. What to do?

Solution:  Make a new ``sources`` object. Thats it.
Your new subclass should have a ``get_spectrum(outwave=[], filters=[],
**params)`` method that converts a dictionary of parameters, a list of filters,
and a wavelength grid into a model SED and spectrum, and returns the spectrum,
the photometry, and any ancillary info. You will have to write that.

See any of the ``sources`` classes for the appropriate ``get_spectrum`` API.


Multiple Spectra
----------------------

We are working on this.
It will involve a new ProspectorParameters subclass that can loop over
``obs['spectra']`` and apportion vectors of parameters correctly.


Linear Algebra
--------------------

This code is slow! Get better math.

If you are fitting spectra with a flexible noise model, this code does lots of
matrix inversions as part of the gaussian process. This is very computationally
intensive, and massive gains can be made by using optimized linear algebra
libraries.



.. |Codename| replace:: Prospector
