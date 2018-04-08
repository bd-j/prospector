Output format
================

By default the output of the code is an HDF5 file, with filename
``<output>_<timestamp>_mcmc.h5``

Optionally several pickle files
(`pickle <https://docs.python.org/2/library/pickle.html>`_ is Python's internal object serialization module),
roughly equivalent to IDL SAVE files, can be output.
These may be convenient, but are not very portable.


HDF5 output
---------------------
The output HDF5 file contains ``datasets`` for the input observational data and the MCMC sampling chains.
A significant amount of metadata is stored as JSON in dataset attributes.
Anything that could not be JSON serialized during writing will have been pickled instead,
with the pickle stored as string data in place of the JSON.

The HDF5 files can read back into python using

.. code-block:: python

		import prospect.io.read_results as pread
		filename = "<outfilestring>_<timestamp>_mcmc.h5"
		results, obs, model = pread.results_from(filename)

which gives a ``results`` dictionary, the ``obs`` dictionary containing the data to which the model was fit,
and the ``model`` object used in the fitting.
The ``results`` dictionary contains
the production MCMC chains from `emcee` or the chains and weights from `dynesty`,
basic descriptions of the model parameters,
and the ``run_params`` dictionary.
Some additional ancillary information is stored, such as code versions, runtimes, MCMC acceptance fractions,
and model parameter positions at various phases of of the code.
There is also a text version of the **parameter file** used.
The results dictionary contains the information needed to regenerate the *sps* object used in generating SEDs.

.. code-block:: python

		sps = pread.get_sps(res)


Pickles
----------------------
The results pickle is relatively portable file, which is a serialization of a dictionary containing

The results pickle is a serialization of the results dictionary,
and has ``<timestamp>_mcmc`` appended onto the output file string specified when the code was run,
where ``timestamp`` is in UT seconds.
It uses only basic scientific python types (e.g. dictionaries, lists, and numpy arrays).
It should therefore be readable on any system with Python and Numpy installed.
This can be accomplished with

.. code-block:: python

		import pickle
		filename = "<outfilestring>_<timestamp>_mcmc"
		with open(filename, "rb") as f:
		    result = pickle.load(f)
		print(result.keys())

The model pickle has the extension ``<timestamp>_model``.
It is a direct serialization of the model object used during fitting, and is thus extremely useful for regenerating posterior samples of the SED,
or otherwise exploring properties of the model.

However, this requires Python and a working |Codename| installation of a version compatible with the one used to generate the model pickle.
If that is possible, then the following code will read the model pickle:

.. code-block:: python

		import pickle
		model_file = "<outfilestring>_<timestamp>_model"
		with open(model_file, 'rb') as mf:
		    mod = pickle.load(mf)
		print(type(mod))

If Powell optimization was performed, this pickle also contains the optimization results (as a list of Scipy OptimizerResult objects).



Basic diagnostic plots
-----------------------------
Several methods for visualization of the results are included in the |Codename|.io.read_results module.

First, the results file can be read into useful dictionaries and objects using

.. code-block:: python

		import prospect.io.read_results as rr
		filename = "<outfilestring>_<timestamp>_mcmc"
		results, obs, model = rr.results_from(filename)

See the help for ``prospect.io.read_results_from()`` for a description of the returned objects.

It is often desirable to plot the parameter traces for the MCMC chains.
That is, one wants to see the evolution of the parameter values as a function of MCMC iteration.
This is useful to check for convergence.
It can be done easily for both `emcee` and `dynesty` results by

.. code-block:: python

		tracefig = rr.traceplot(results)

Another useful thing is to look at the "corner plot" of the parmeters.
If one has the `corner.py (https://github.com/dfm/corner.py)`_ package, then

.. code-block:: python

		cornerfig = rr.subcorner(results, showpars=mod.theta_labels()[:5])

will return a corner plot of the first 5 free parameters of the model.
If ``showpars`` is omitted then all free parameters will be plotted.
There are numerous other options to the ``subcorner`` method, which is a thin wrapper on `corner.py`,
but they are documented (``help(rr.subcorner)``)

Finally, one often wants to look at posterior samples in the space of the data, or perhaps the maximum a posteriori parameter values.
Taking the MAP as an example, this would be accomplished by

.. code-block:: python

		import np

		# Find the index of the maximum a posteriori sample (for `emcee` results)
		ind_max = results["lnprobability"].argmax()
		walker, iteration = np.unravel_index(ind_max, results["lnprobability"].shape)
		theta_max = results["chain"][walker, iteration, :]

		# We need the SPS object to generate a model
		sps = rr.get_sps(results)
		# now generate the SED for the max. a post. parameters
		spec, phot, x = model.mean_model(theta_max, obs=obs, sps=sps)

		# Plot the data and the MAP model on top of each other
		import matplotlib.pyplot as pl
		if obs['wave'] is None:
		    wave = sps.wavelengths
		else:
		    wave = obs['wavelength']
		pl.plot(wave, obs['spectrum'], label="Data")
		pl.plot(wave, spec, label="MAP model")


.. |Codename| replace:: Prospector
