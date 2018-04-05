Tutorial
============

Here is a quick demo of how to get up and running with |Codename|.

We assume you have installed |Codename| and all its dependencies as laid out in the docs.
The next thing you need to do is make a temporary work directory, ``<workdir>``

.. code-block:: shell
		
		cd <workdir>
		cp <codedir>/scripts/prospector*.py .
		cp <codedir>/demo/demo_* .

We now have a prospector executable, a *parameter file*  or two, and some data.
Take a look at the data file in an editor, you'll see it is a simple ascii file, with a few rows and several columns.
Each row is a different galaxay, each column is a different piece of information about that galaxy.

This is just an example.
In practice |Codename| can work with a wide variety of data types.

Open up ``demo_params.py`` in an editor, preferably one with syntax highlighting.
You'll see that it's a python file.
Some things are imported, and then there is the ``run_params`` dictionary.
This dictionary is where you put variables that control the operation of the code.
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
This one is pretty straightforward, it simply instantiates our CSPBasis object.
After that is ``load_gp``, which is for complexifying the noise model -- ignore that for now.

Now on to the fun part.
The ``load_model`` function is where the model that we will fit will be constructed.
First we have to specify a dictionary or list of model parameter specifications (see models.rst).
Each specification is a dictionary that describes a single parameter.
We can build the model from predefined sets of model parameter specifications,
stored in the ``models.templates.TemplateLibrary`` directory.
You'll note that for 5 of these parameters we have set.
Any parameters with ``"isfree": True`` in its specification will be varied during the fit.
We have set priors on these parameters, including prior arguments.
Any free parameter *must* have an associated prior.
Other parameters have their value set (to the value of the ``"init"`` key) but do not vary during the fit.
They can be made to vary by setting ``"isfree": True`` and specifying a prior.
Parameters not listed here will be set to their default values.
For CSPBasis this means the default values in the ``fsps.StellarPopulation()`` object,
see `python-fsps (http://dan.iel.fm/python-fsps/current/)`_ for details

Finally, the ``load_model()`` function takes the ``model_params`` collection  and
uses it to instantiate a ``SedModel`` object.
If you wanted to change the specification of the model using command line arguments,
you could do it in this function using keyword arguments that are also keys of ``run_params``.
This can be useful for example to set the initial value of the redshift ``"zred"`` on an object-by-object basis.

Running a fit
----------------------

To run this fit on object 0, we would do the following at the command line

.. code-block:: shell
		
		python prospector.py --param_file=demo_params.py --objid=0 --outfile=demo_obj0

If we wanted to change something about the MCMC parameters, we could also do that at the command line

.. code-block:: shell
		
		python prospector.py --param_file=demo_params.py --objid=0 --outfile=demo_obj0 \
		--nwalkers=32 --niter=1024

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
``demo_obj0_<timestamp>_mcmc.h5``. 
This is an HDF5 file containing sampling results and various configuration data,
as well as the observational data that was fit.
By setting ``run_params["output_pickles"]=True`` you can also output versions of this information in the less portable pickle format.
We will read the HDF5 with python and make some plots using utilities in |Codename|

To read the data back in from the output files that we've generated, use
methods in ``prospect.io.read_results``.  There are also some methods in this
module for basic diagnostic plots. The ``subcorner`` method requires that you have the `corner
<http://corner.readthedocs.io/en/latest/>`_ package installed.

.. code-block:: python
		
		import prospect.io.read_results as pread
		res, obs, mod = pread.results_from("demo_obj_<timestamp>_mcmc.h5")
		tracefig = pread.traceplot(res)
		cornerfig = pread.subcorner(res, start=0, thin=5)

The ``res`` object is a dictionary containing various useful results.
You can look at ``res.keys()`` to see a list of what it contains.
The ``obs`` object is just the ``obs`` dictionary that was used in the fitting.
The ``mod`` object is the model object that was used in the fitting.
There are also numerous more or less poorly documented convenience methods in
the ``prospect.utils.plotting``.

If necessary, one can regenerate models at any position in the posterior chain.  This requires that we have the sps object used in the fitting to generate models, which we can regenerate using the ``read_results.get_sps()`` method

.. code-block:: python
		
		import prospect.io.read_results as pread
		res, obs, mod = pread.results_from("demo_obj_<timestamp>_mcmc")
		# We need the correct sps object to generate models
		sps = pread.get_sps(res)

Now we will choose a specific parameter value from the chain, using the last iteration of the first walker, and plot what the observations and the model look like, as well as the uncertainty normalized residual

.. code-block:: python
		
		# Choose the walker and iteration number,
		# if you used emcee for the inference
		walker, iteration = 0, -1
		# Get the modeled spectra and photometry.
		# These have the same shape as the obs['spectrum'] and obs['maggies'] arrays.
		spec, phot, mfrac = mod.mean_model(res['chain'][walker, iteration, :], obs=res['obs'], sps=sps)
		# mfrac is the ratio of the surviving stellar mass to the formed mass (the ``"mass"`` parameter).
		# Plot the model SED
		import matplotlib.pyplot as pl
		wave = [f.wave_effective for f in res['obs']['filters']]
		sedfig, sedax = pl.subplots()
		sedax.plot(wave, res['obs']['maggies'], '-o', label='Observations')
		sedax.plot(wave, phot, '-o', label='Model at {},{}'.format(walker, iteration))
		sedax.set_ylabel("Maggies")
		sedax.set_xlabel("wavelength")
		# Plot residuals for this walker and iteration
		chifig, chiax = pl.subplots()
		chi = (res['obs']['maggies'] - phot) / res['obs']['maggies_unc']
		chifig.plot(wave, chi, 'o')
		chiax.set_ylabel("Chi")
		chiax.set_xlabel("wavelength")


.. |Codename| replace:: Prospector
