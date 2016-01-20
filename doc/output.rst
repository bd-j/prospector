Output format
================

The output of the code is in several files.

Two of these are pickle files (`pickle <https://docs.python.org/2/library/pickle.html>`_
is Python's internal object serialization module), roughly equivalent to IDL SAVE files.
The are included for convenience, but are not very portable.

The third file is an HDF5 file, described below.

Results pickle
----------------------
The results pickle is relatively portable file, which is a serialization of a dictionary containing
the production MCMC chains from emcee,
the input ``obs`` dictionary,
basic descriptions of the model parameters,
and the ``run_params`` dictionary.
Some additional ancillary information is stored, such as code versions, runtimes, MCMC acceptance fractions,
and model parameter positions at various phases of of the code.
There is also a text version of the **parameter file** used.

The results pickle has ``<timestamp>_mcmc`` appended onto the output file string specified when the code was run,
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

Model pickle
----------------------
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

HDF5 output
---------------------
The output HDF5 file contains datasets for the input observational data and the MCMC sampling chains
A significant amount of metadata is stored as JSON in dataset attributes.


Basic diagnostic plots
-----------------------------
Several methods for visualization of the results are included in the |Codename|.read_results module.

First, the results and model pickles can be read using

.. code-block:: python

		import bsfh.read_results as bread
		filename = "<outfilestring>_<timestamp>_mcmc"
		results, model, powell_results = bread.read_from(filename)

It is often desirable to plot the parameter traces for the MCMC chains.
That is, one wants to see the evolution of the parameter values as a function of MCMC iteration.
This is useful to check for convergence.
It can be done easily (if ugly) by

.. code-block:: python

		efig = bread.param_evol(results)

Another useful thing is to look at the "corner plot" of the parmeters.
If one has the `corner.py (https://github.com/dfm/corner.py)`_ package, then 

.. code-block:: python

		cfig = bread.subtriangle(results, showpars=mod.theta_labels()[:5])

will return a corner plot of the first 5 parameters of the model.  If ``showpars`` is omitted then all parameters will be plotted.  There are numerous other options to the ``subtriangle`` method, but they are documented (``help(bread.subtriangle)``)

Finally, one often wants to look at posterior samples in the space of the data, or perhaps the maximum a posteriori parameter values.
Taking the MAP as an example, this would be accomplished by

.. code-block:: python

		import np
		obs = results["obs"]

		# Find the index of the maximum a posteriori
		ind_max = results["lnprobability"].argmax()
		walker, iteration = np.unravel_index(ind_max, results["lnprobability"].shape)
		theta_max = results["chain"][walker, iteration, :]

		# We need the SPS object to generate a model
		from bsfh.models import model_setup
		sps = model_setup.load_sps(**results["run_params"])
		# now generate the SED for the max. a post. parameters
		spec, phot, x = model.mean_model(theta_max, obs=obs, sps=sps)

		# Plot the data and the MAP model on top of each other
		import matplotlib.pyplot as pl
		pl.plot(obs['wavelength'], obs['spectrum'], label="Data")
		pl.plot(obs['wavelength'], spec, label="MAP model")
