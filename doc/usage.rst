User Interaction
================

The primary user interaction is through the ``prospector_*.py`` scripts and a **parameter file**,
a python file in which several variables and functions must be defined.
Command line syntax calls the ``prospector.py`` script as an executable and is as follows for single thread execution:

.. code-block:: shell

		python prospector.py --param_file=parameter_file.py

Additional command line options can be given (see below).
You can copy the ``prospector.py`` script to wherever you intend to run the code, or put it in your path.

The parameter file
-------------------------

The parameter ile is a python file, and can have whatever name you wish.
The required variables and functions in a **parameter file** are:

1. ``run_params``:
   This dictionary specifies global options and parameters used for setting up
   and running the code. It is passed as keywords to all the functions listed below.

2. :py:meth:`load_obs`: 
   This function will take the ``run_params`` dictionary as keyword arguments
   and returns on obs dictionary (see :doc:`dataformat` .)

3. :py:meth:`load_model`:
   This function will take the ``run_params`` dictionary as keyword arguments
   and return an instance of a :class:`ProspectorParams` subclass, containing
   information about the parameters of the model (see :doc:`models` .)

4.  :py:meth:`load_sps`:
    This function will take the ``run_params`` dictionary as keyword arguments
    and return an **sps** object, which must have the method
    :py:meth:`get_spectrum` defined.  This object generally includes all the
    spectral libraries necessary to build a model, as well as much of the model
    building code and as such has a large memory footprint.

5.  :py:meth:`load_gp`:
    This function should return a :py:class:`NoiseModel` object for the spectroscopy and/or
    photometry.  Either or both can be ``None`` in which case the likelihood
    will not include covariant noise and is equivalent to basic :math:`\chi^2`.


The ``run_params`` dictionary
-----------------------------------------

The following parameters conrol key aspects of the operation of the code,
and are stored in a special dictionary called ``run_params``.
The ``run_params`` dictionary is passed as keyword arguments to the
:py:meth:`load_obs`, :py:meth:`load_sps`, and :py:meth:``load_model`
functions.
This means you can add to it if you want additional parameters to control the
output of these functions.

General parameters:

``"verbose"``
    Boolean.
    If ``True`` lots of diagnostic information will be written to stdout during execution.

``"debug"``
    Boolean.  If ``True``, halt before starting minimization or sampling.
    Can be useful to debug inputs when used with an interactive python session.

``"outfile"``
    String.  Base name of the output files.
    Various extensions as well as a time stamp will be appened to this string.

``"output_pickles"``
   Boolean.  If ``True`` then write pickle files in addition to HDF5.  Deafult
   is ``False``.

Nested sampling parameters:

``"dynesty_npoints"``
    The number of active points in the nested sampling algorithm, defaults to 200

Ensemble sampling parameters:

``"nwalkers"``
    Integer.  Number of emcee walkers.

``"nburn"``
    List of integers, e.g. ``[32, 64, 64]`` giving the number of iterations in each burn-in run.
    After each number of iterations the walkers will be trimmed and a new ball of
    samplers will be initialized around the highest-probability walker.
    This can help avoid stuck walkers and speed up burn-in.

``"niter"``
    Integer.  Number of iterations for the final production run.

``"initial_disp"``
    Float.  Default value to use for the dispersion in the parameter

``"interval"``
		A number between 0 and 1 giving the fractional interval at which to
		incrementally save the chain to disk.  This can be helpful if there is a
		possibility that your process might be killed but you don't want to lose all
		the hard-won sampling that has taken place so far.

Optimization parameters:

``"do_powell"``
    Boolean.  If ``True``, do a round of Powell minimization before MCMC sampling.
    If MPI is enabled then ``np`` minimizations from different initial conditions will be run,
    and the highest likelihood result chosen as the center for the sampler ball.
		This can perform poorly if there are many very degenerate parameters,
		or if the parameter scales are very different.

``"ftol"``
    Float.  For the Powell minimization.

``"maxfev"``
    Integer.  For the Powell minimization.

``"do_levenburg"``
		Boolean.   If ``True``, do a round of Levenburg-Marquardt least-squares optimization before MCMC sampling.
		Requires ``"do_powell": False``

``"nmin"``
		Number of starting conditions to sample from the prior for use in L-M optimization.
		The initial value taken from the model_params dict is always included as one of the starting conditions.
		The best final position is chosen from all optimizations.
		This provides some robustness against local minima.

Data manipulation parameters:

``"logify_data"``
    optional Boolean.  Switch to do the fitting in log flux space.
    Not recommended, as it distorts your errors.

``"rescale_spectrum"``
    Boolean.  If ``True``, rescale the spectrum to have an average of 1 before doing anything.
    The scaling parameter is stored in the ``obs`` dict as ``obs["rescale"]``.
    This parameter should be ``False`` unless you are simultaneously fitting
    photometry (see ``normalize_spectrum`` below), or you are fitting for the
    spectral calibration as well.

``"normalize_spectrum"``
    optional Boolean.
    If ``True`` make an initial guess of the relative normalization of the spectrum and the photometry,
    using synthetic photometry of the spectrum through the filter specified ``"norm_band_name"``.
    The normalization guess is stored in the obs dictionary (as ``normalization_guess``).

``"norm_band_name"``
    String.  Name of the filter to use for making an initial guess at the spectral normalization.

Source Basis Parameters:

``"zcontinuous"``
    Integer.  If fitting galaxy spectra using py-FSPS, this is passed to the StellarPopulation
    object on instantiation and controls how metallicity interpolation is done.
    See the python-FSPS documentation for details.

``"libname"``
   String.  If fitting stellar spectra, this is the name of the HDF5 file containing the stellar spectral grid.


There is limited support for command line overrides of the ``run_params`` dictionary values.
For example

.. code-block:: shell

		python prospector.py –-param_file=demo_params.py –-nwalkers=128``

will cause the code to use 128 walkers regardless of the value given directly in the ``run_params`` dictionary.
Such overriden parameters must be present as keys in the ``run_params`` dictionary,
as they will be coerced to have the same data type as the default value in the ``run_params`` dictionary.
Currently only scalars can be changed at the command line.
