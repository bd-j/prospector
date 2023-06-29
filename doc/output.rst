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

		import prospect.io.read_results as reader
		filename = "<outfilestring>_<timestamp>_mcmc.h5"
		results, obs, model = reader.results_from(filename)

which gives a ``results`` dictionary, the ``obs`` dictionary containing the data to which the model was fit,
and the ``model`` object used in the fitting.
The ``results`` dictionary contains
the production MCMC chains from `emcee` or the chains and weights from `dynesty`,
basic descriptions of the model parameters,
and the ``run_params`` dictionary.
Some additional ancillary information is stored, such as code versions, runtimes, MCMC acceptance fractions,
and model parameter positions at various phases of of the code.
There is also a string version of the **parameter file** used.
The results dictionary contains the information needed to regenerate the *sps* object used in generating SEDs.

.. code-block:: python

		sps = reader.get_sps(res)

It can sometimes be difficult to reconstitute the model object if it is
complicated, for example if it was built by referencing files or data that are
no longer available. For this reason it is suggested that references to
filenames in parameter files be made through command-line arguments that can be
altered easily when reconsitituting the model.


Basic diagnostic plots
-----------------------------
For detailed plotting, see the :py:mod:`prospect.plotting` module.
Several methods for basic visualization of the results are also included in the :py:mod:`prospect.io.read_results` module.

First, the results file can be read into useful dictionaries and objects using :py:meth:prospect.io.read_results.results_from``

.. code-block:: python

		import prospect.io.read_results as reader
		filename = "<outfilestring>_<timestamp>_mcmc"
		results, obs, model = reader.results_from(filename)

It is often desirable to plot the parameter traces for the MCMC chains.
That is, one wants to see the evolution of the parameter values as a function of MCMC iteration.
This can be useful to check for convergence.
It can be done easily for both `emcee` and `dynesty` results by

.. code-block:: python

		tracefig = reader.traceplot(results)

Another useful thing is to look at the "corner plot" of the parmeters.
If one has the `corner.py <https://github.com/dfm/corner.py>`_ package, then

.. code-block:: python

		cornerfig = reader.subcorner(results, showpars=model.theta_labels()[:5])

will return a corner plot of the first 5 free parameters of the model.
If ``showpars`` is omitted then all free parameters will be plotted.
There are numerous other options to the :py:meth:`prospect.io.read_results.subcorner` method, which is a thin wrapper on `corner.py`.

Finally, one often wants to look at posterior samples in the space of the data, or perhaps the maximum a posteriori parameter values.
Taking the MAP as an example, this would be accomplished by

.. code-block:: python

        import numpy as np
        # Find the index of the maximum a posteriori sample
        ind_max = results["lnprobability"].argmax()
        if res["chain"].ndim > 2:
            # emcee
            walker, iteration = np.unravel_index(ind_max, results["lnprobability"].shape)
		    theta_max = results["chain"][walker, iteration, :]
        elif res["chain"].ndim == 2:
            # dynesty
            theta_max = results["chain"][indmax, :]

        # We need the SPS object to generate a model
        sps = reader.get_sps(results)
        # now generate the SED for the max. a post. parameters
        spec, phot, x = model.predict(theta_max, obs=obs, sps=sps)

        # Plot the data and the MAP model on top of each other
        import matplotlib.pyplot as pl
        if obs['wave'] is None:
		    wave = sps.wavelengths
        else:
            wave = obs['wavelength']
        pl.plot(wave, obs['spectrum'], label="Spec Data")
        pl.plot(wave, spec, label="MAP model spectrum")
        if obs['filters'] is not None:
            pwave = [f.wave_effective for f in obs["filters"]]
            pl.plot(pwave, obs['maggies'], label="Phot Data")
            pl.plot(pwave, phot, label="MAP model photometry")


However, if all you want is the MAP model this may be stored for you,
without the need to regenerate the ``sps`` object

.. code-block:: python

        import matplotlib.pyplot as pl
		best = res["bestfit"]
        a = model.params["zred"] + 1
        pl.plot(a * best["restframe_wavelengths"], best['spectrum'], label="MAP spectrum")
        if obs['filters'] is not None:
            pwave = [f.wave_effective for f in obs["filters"]]
            pl.plot(pwave, best['photometry'], label="MAP photometry")



.. |Codename| replace:: Prospector
