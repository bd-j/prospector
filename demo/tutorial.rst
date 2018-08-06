Tutorial
============

Here is a quick guide to get up and running with |Codename|.

We assume you have installed |Codename| and all its dependencies as laid out in the docs.
The next thing you need to do is make a temporary work directory, ``<workdir>``

.. code-block:: shell
		
		cd <workdir>
		cp <codedir>/scripts/prospector*.py .
		cp <codedir>/demo/demo_* .

We now have some prospector executable scripts, a *parameter file*  or two, and some data.
Take a look at the ``demo_photometry.dat`` file in an editor, you'll see it is a simple ascii file, with a few rows and several columns.
Each row is a different galaxay, each column is a different piece of information about that galaxy.

This is just an example.
In practice |Codename| can work with a wide variety of data types.

The parameter file
----------------------

Open up ``demo_params.py`` in an editor, preferably one with syntax highlighting.
You'll see that it's a python file.
Some things are imported, and then there is the ``run_params`` dictionary.
This dictionary is where you store variables that control the operation of the code.
It is passed to each of the other main setup functions in ``param_file.py``

About those imports.
Since we are fitting galaxies with a composite stellar population,
we made sure to import the ``sources.CSPSpecBasis`` class.
If you were fitting stars or non-parameteric SFHs you would use a different
object from the ``sources`` module.

The next thing to look at is the ``load_obs()`` function.
This is where you take the data from whatever format you have and
put it into the format required by |Codename| for a single object.
This means you will have to modify this function heavily for your own use.
But it also means you can use your existing data formats.

Right now, the ``load_obs`` function just reads ascii data from a file,
picks out a row (corresponding to the photometry of a single galaxy),
and then makes a dictionary using data in that row.
You'll note that both the datafile name and the object number are keyword arguments to this function.
That means they can be set at execution time on the command line,
by also including those variables in the ``run_params`` dictionary.
We'll see an example later.

When you write your own ``load_obs`` function, you can add all sorts of keyword arguments that control its output
(for example, an object name or ID number that can be used to choose or find a single object in your data file).
You can also import helper functions and modules.
These can be either things like astropy, h5py, and sqlite or your own project specific modules and functions.
As long as the output dictionary is in the right format (see dataformat.rst), the body of this function can do anything.

Ok, now we go to the ``load_sps`` function.
This one is pretty straightforward, it simply instantiates our ``CSPSpecBasis`` object.
After that is ``load_gp``, which is for complexifying the noise model -- ignore that for now.

Now on to the fun part.
The ``load_model`` function is where the model that we will fit will be constructed.
The specific model that you choose to construct depends on your data and your scientific question.
First we have to specify a dictionary or list of model parameter specifications (see models.rst).
Each specification is a dictionary that describes a single parameter.
We can build the model from predefined sets of model parameter specifications,
stored in the ``models.templates.TemplateLibrary`` directory.
In this example we choose the ``"parameteric"`` set, which has the parameters necessary for a delay-tau SFH fit.
This parameter set can be inspected in any of the following ways

.. code-block:: python
		
		from prospect.models.templates import TemplateLibrary
		# Show basic descriptin of all pre-defined parameter sets
		TemplateLibrary.show_contents()
		# method 1: print the whole dictionary of dictionaries
		print(TemplateLibrary["parametric_sfh"])
		# Method 2: show a summary of the free and fixed parameters
		print(TemplateLibrary.describe("parametric_sfh")

You'll see that this model has 5 free parameters.
Any parameters with ``"isfree": True`` in its specification will be varied during the fit.
We have set priors on these parameters, including prior arguments.
Any free parameter *must* have an associated prior.
Other parameters have their value set (to the value of the ``"init"`` key) but do not vary during the fit.
They can be made to vary by setting ``"isfree": True`` and specifying a prior.
Parameters not listed here will be set to their default values.
For ``CSPSpecBasis`` this means the default values in the ``fsps.StellarPopulation()`` object,
see `python-fsps <http://dan.iel.fm/python-fsps/current/>`_ for details
Once you get a set of parameters from the ``TemplateLibrary`` you can modify or add parameter specifications.

Finally, the ``load_model()`` function takes the ``model_params`` dictionary or list that you build and
uses it to instantiate a ``SedModel`` object.
If you wanted to change the specification of the model using command line arguments,
you could do it in this function using keyword arguments that are also keys of ``run_params``.
This can be useful for example to set the initial value of the redshift ``"zred"`` on an object-by-object basis.

Running a fit
----------------------
There are two kinds of fitting packages that can be used with |Codename|.
The first is ``emcee`` which implements ensemble MCMC sampling,
and the second is ``dynesty``, which implements dynamic nested sampling.
Choosing which to use involves choosing which script to run

To run this fit on object 0 using ``emcee``, we would do the following at the command line

.. code-block:: shell
		
		python prospector.py --param_file=demo_params.py --objid=0 \
                --outfile=demo_obj0_emcee 

If we wanted to change something about the MCMC parameters, or fit a different object,
we could also do that at the command line

.. code-block:: shell
		
		python prospector.py --param_file=demo_params.py --objid=1 \
		--outfile=demo_obj1_emcee --nwalkers=32 --niter=1024

And if we want to use nested sampling with ``dynesty`` we would do the following

.. code-block:: shell
		
		python prospector_dynesty.py --param_file=demo_params.py --objid=0 \
		--outfile=demo_obj0_dynesty 

Finally, it is sometimes useful to run the script from the interpreter to do some checks.
This is best done with the IPython enhanced interactive python.

.. code-block:: shell
		
		ipython
		In [1]: %run prospector.py --param_file=demo_params.py --objid=0 --debug=True

The ``--debug=True`` flag will halt execution just before the fitting starts.
You can then inspect the ``obsdat`` dictionary, the ``model`` object,
and the ``run_params`` dictionary to make sure everything is working fine.

Working with the output
--------------------------------
After the fit is completed we should have a file with a name like
``demo_obj0_<fitter>_<timestamp>_mcmc.h5``. 
This is an HDF5 file containing sampling results and various configuration data,
as well as the observational data that was fit.
By setting ``run_params["output_pickles"]=True`` you can also output versions of this information in the less portable pickle format.
We will read the HDF5 with python and make some plots using utilities in |Codename|

To read the data back in from the output files that we've generated, use
methods in ``prospect.io.read_results``. 

.. code-block:: python
		
		import prospect.io.read_results as pread
		res, obs, mod = pread.results_from("demo_obj_<fitter>_<timestamp>_mcmc.h5")

The ``res`` object is a dictionary containing various useful results.
You can look at ``res.keys()`` to see a list of what it contains.
The ``obs`` object is just the ``obs`` dictionary that was used in the fitting.
The ``mod`` object is the model object that was used in the fitting.

There are also some methods in this module for basic diagnostic plots.
The ``subcorner`` method requires that you have the `corner
<http://corner.readthedocs.io/en/latest/>`_ package installed.
It's possible now to examine the traces (i.e. the evolution of parameter value with MC iteration)
and the posterior PDFs for the parameters.

.. code-block:: python

		# Trace plots
		tfig = pread.traceplot(res)
		# Corner figure of posterior PDFs
		cfig = pread.subcorner(res)

If you want to get the `maximum a. posteriori` values, or percentiles of the posterior pdf,
that can be done as follows
(note that for ``dynesty`` the weights of each posterior sample must be taken into account when calculating quantiles)
:

.. code-block:: python

		# Maximum posterior probability sample
		imax = np.argmax(res['lnprobability'])
		csz = res["chain"].shape
		if res["chain"].ndim > 2:
		    # emcee
		    i, j = np.unravel_index(imax, res['lnprobability'].shape)
		    theta_max = res['chain'][i, j, :].copy()
		    flatchain = res["chain"].reshape(csz[0] * csz[1], csz[2])
		else:
		    # dynesty
		    theta_max = res['chain'][imax, :].copy()
		    flatchain = res["chain"]

		# 16th, 50th, and 84th percentiles of the posterior
		from prospect.utils.plotting import quantile
		post_pcts = [quantile(flatchain[:, i], percents=[16, 50, 84],
		                                    weights=res.get("weights", None))
				      for i in range(mod.ndim)]

If necessary, one can regenerate models at any position in the posterior chain.
This requires that we have the sps object used in the fitting to generate models, which we can regenerate using the ``read_results.get_sps()`` method.

.. code-block:: python
		
		# We need the correct sps object to generate models
		sps = pread.get_sps(res)

Now we will choose a specific parameter value from the chain and plot what the observations and the model look like, as well as the uncertainty normalized residual.  For ``emcee`` results we will use the last iteration of the first walker, while for ``dynesty`` results we will just use the last sample in the chain.

.. code-block:: python
		
		# Choose the walker and iteration number,
		if res["chain"].ndim > 2:
 		    # if you used emcee for the inference
		    walker, iteration = 0, -1
		    theta = res['chain'][walker, iteration, :]
		else:
		    # if you used dynesty
		    theta = res['chain'][iteration, :]

		# Get the modeled spectra and photometry.
		# These have the same shape as the obs['spectrum'] and obs['maggies'] arrays.
		spec, phot, mfrac = mod.mean_model(theta, obs=res['obs'], sps=sps)
		# mfrac is the ratio of the surviving stellar mass to the formed mass (the ``"mass"`` parameter).

		# Plot the model SED
		import matplotlib.pyplot as pl
		wave = [f.wave_effective for f in res['obs']['filters']]
		sedfig, sedax = pl.subplots()
		sedax.plot(wave, res['obs']['maggies'], '-o', label='Observations')
		sedax.plot(wave, phot, '-o', label='Model at {},{}'.format(walker, iteration))
		sedax.set_ylabel("Maggies")
		sedax.set_xlabel("wavelength")
		sedax.set_xscale('log')

		# Plot residuals for this walker and iteration
		chifig, chiax = pl.subplots()
		chi = (res['obs']['maggies'] - phot) / res['obs']['maggies_unc']
		chiax.plot(wave, chi, 'o')
		chiax.set_ylabel("Chi")
		chiax.set_xlabel("wavelength")
		chiax.set_xscale('log')


.. |Codename| replace:: Prospector
