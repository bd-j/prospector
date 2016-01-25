Advanced Usage
==============

Parameter transfomations
------------------------

Problem: But I don't want to sample in ``dumb_parameter``!
Solution: Transform to parameters that are easier to sample in.

This can be done by making ``dumb_parameter`` fixed (``"isfree" = False``) and adding another key to the parameter description, ``"depends_on"``.
The value of ``"depends_on"`` is a function which takes as arguments all the model parameters and returns the value of ``dumb_parameter``.
For example:
.. code-block:: python

		def delogify(logtau=0, **extras):
		    return 10**logtau

could be used to set the value of ``tau`` using the sampling parameter ``logtau``
(i.e., sample in the log of a parameter, though setting a logarithmic prior is equivalent in terms of the posterior).
This dependency function must take optional extra keywords (``**extras``) because the entire parameter dictionary will be passed to it.
Then add another parameter ``smart_parameter`` to ``model_list`` that can vary (and upon which ``dumb_parameter`` depends).

This pattern can also be used to tie arbitrary parameters together (e.g. gas-phase and stellar metallicity) while still allowing them to vary.
A parameter may depend on multiple other parameters, and several parameters may depend on a single other parameter.

It is important that any parameter with the ``"depends_on"`` key is fixed.
For portability and easy reconstruction of the model is important that the ``depends_on`` function be defined within the parameter file.

User defined models
-------------------

Problem: The pre-packaged models suck! You can do better.
Or, you have stars instead of stellar populations. Or spectra of the IGM or planets or AGN or something.
What to do?

Solution:  Make a new ``source_basis`` object. Thats it.
Your new subclass should have a ``get_spectrum(outwave=[], filters=[], **params)`` method that
converts a dictionary of parameters, a list of filters, and a wavelength grid into a model SED and spectrum,
and returns the spectrum, the photometry, and any ancillary info.
You will have to write that.
See any of the ``source_basis`` objects for the appropriate ``get_spectrum`` API.
Note that ``source_basis.StarBasis`` and ``source_basis.BigStarBasis`` are fairly general objects for grid storage and interpolation, feel free to subclass them if you are using grids of SEDs that can be stored in HDF5 files.

Multiple Spectra
---------------

We are working on this.
It will involve a new ThetaParameters subclass that can loop over ``obs['spectra']`` and apportion vectors of parameters correctly.

Linear Algebra
--------------

This code is slow! Get better math.

If you are fitting spectra with a flexible noise model,
this code does lots of matrix inversions as part of the gaussian process.
This is very computationally intensive, and massive gains can be made by using optimized linear algebra libraries.

MPI
---

This code is slow! Get moar processors.

Install some kind of MPI on your system (openMPI, mpich2, mvapich2),
make sure mpi4py is also installed against this MPI installation,
and use the syntax
``mpirun -np <N> python prospectr.py –param_file=<param_file>``

This causes likelihood evaluations for different walkers to be made in parallel.
For optimal results, the number of emcee walkers should be :math:`N*(np-1)`,
where N is an even integer and :math:`np` is the number of available processors.

Note that specific MPI implementations may have different mpirun commands, or
may require that python-mpi be called instead of just python.

Noise Modeling
--------------

This is handled by specifiying rules for constructing a covariance matrix, and supplying a ``load_gp()`` method in the parameter file.
Flexibility in this is an active area of code development.

Mock data
---------

Really this should not be advanced.
Everyone should do mock data tests.
So we are trying to make it easy.
See demo/demo_mock_params.py for a suggestion, especially the ``load_obs()`` function.
