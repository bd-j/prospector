User Interaction
================

The primary user interaction is through a **parameter file**,
a python file in which several variables and functions must be defined.
Command line syntax calls the prospector script as an executable and
is as follows for single thread execution:

.. code-block:: shell
		
		cd demo
		python prospector.py –param_file=demo_params.py

Additional command line options can be given (see below).
The required variables and functions in ``demo_params.py`` are

- ``run_params``: A dictionary.
  This dictionary specifies global options and parameters used for setting up and running the code.

- ``model_params``: A list of dictionaries.
  Each element is a dictionary discribing an individiual model parameter,
  to be varied or kept constant during the fitting

- ``load_obs``: A function.
  This function will take the ``run_params`` distionary as keyword arguments and returns on obs dictionary (see Data_)

- ``load_model``: A function

- ``load_sps``: A function

See ``tests.sh`` and the parameter files and prospectr.py script in the
“demo” directory of the repository for more examples of proper usage.

The ``run_params`` dictionary
----------------------------

The following parameters conrol key aspects of the operation of the
code, and are stored in a special dictionary called ``run_params``:

General parameters:

-  ``"verbose"`` Boolean

-  ``"debug"`` Boolean.  If ``True``, halt before starting minimization or sampling.
   Can be useful to debug inputs when used with an interactive python session.

-  ``"outfile"`` string.  Base name of the output files.
   Various extensions as well as a time stamp will be appened to this string.

Fitter parameters:

-  ``"nwalkers"`` integer.  Number of emcee walkers.

-  ``"nburn"`` list of integers, e.g. ``[32, 64, 64]``
   Number of iterations in each burn-in run.  After each number of iterations the walkers will be
   trimmed and a new ball of samplers will be initialized around the highest-probability walker.
   This can help avoid stuck walkers and speed up burn-in.

-  ``"niter"`` integer.  Number of iterations for the final production run.

-  ``"initial_disp"`` float

-  ``"do_powell"`` Boolean.  If ``True``, do a round of Powell minimization before MCMC sampling.
   If MPI is enabled then ``np`` minimizations from different initial conditions will be run,
   and the highest likelihood result chosen as the center for the sampler ball.

-  ``"ftol"`` float.  For the Powell minimization.

-  ``"maxfev"`` integer.  For the Powell minimization.

Data manipulation parameters:
   
-  ``"logify_data"`` optional Boolean

-  ``"rescale"``  Boolean.  If ``True``, rescale the spectrum to have an average of 1.  Recommended.

-  ``"normalize_spectrum"`` optional Boolean.  If ``True``,
   make an intial guess of the relative normalization of the spectrum and the photometry,
   using synthetic photometry of the spectrum through the filter specified ``"norm_band_name"``.

-  ``"norm_band_name"`` string.  Name of the filter to use for making an  initial guess at
   the spectral normalization.

-  ``"filename"`` string

The run params dictionary is passed as keyword arguments to the ``load_obs``, ``load_sps``, and ``load_model`` functions.
This means you can add to it if you want additional parameters to control the output of these functions.
There is limited support for command line overrides of the ``run_params`` dictionary values.
For example

.. code-block:: shell
		
		python prospectr.py –param_file=demo_params.py –nwalkers=128``

will cause the code to use 128 walkers regardless of the value given directly in the ``run_params`` dictionary.
Such overriden parameters must be present as keys in the ``run_params`` dictionary,
as they will be coerced to have the same data type as the default value in the ``run_params`` dictionary.
Currently only scalars can be changed at the command line.
