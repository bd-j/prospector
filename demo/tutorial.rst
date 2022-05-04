.. _tutorial:

Tutorial
========

Here is a guide to running |Codename| fits from the command line using parameter
files, and working with the output.  This is a generalization of the techniques
demonstrated in the quickstart, with more detailed descriptions of how each of
the ingredients works.

We assume you have installed |Codename| and all its dependencies as laid out in
the docs. The next thing you need to do is make a temporary work directory,
``<workdir>``

.. code-block:: shell

		cd <workdir>
		cp <codedir>/demo/demo_* .

We now have a *parameter file*  or two, and some data. Take a look at the
``demo_photometry.dat`` file in an editor, you'll see it is a simple ascii file,
with a few rows and several columns. Each row is a different galaxy, each column
is a different piece of information about that galaxy.

This is just an example.
In practice |Codename| can work with a wide variety of data types.

The parameter file
------------------

Open up ``demo_params.py`` in an editor, preferably one with syntax
highlighting. You'll see that it's a python file. It includes some imports, a
number of methods that build the ingredients for the fitting, and then an
executable portion.


**Executable Script**

The executable portion of the parameter file that comes after the ``if __name__
== "__main__"`` line is run when the parameter file is called. Here the possible
command line arguments and their default values are defined, including any
custom arguments that you might add. In this example we have added several
command line arguments that control how the data is read and how the model is
built. The supplied command line arguments are then parsed and placed in a
**configuration** dictionary. This dictionary is passed to all the ingredient
building methods (described below), which return the required
:py:class:`Observation` objects and necessary model objects. The data and model
objects are passed to a function that runs the prospector fit
(:py:func:`prospect.fitting.fit_model`). Finally, the fit results are written to
an output file.


**Building the fit ingredients: build_model**

Several methods must be defined in the parameter file to build the ingredients
for the fit. The purpose of these methods and their required output are
described here. You will want to modify some of these for your specific model
and data. Note that each of these functions will be passed a dictionary of
command line arguments. These command line arguments, including any you add to
the command line parser in the executable portion of the script, can therefore
be used to control the behaviour of the ingredient building functions. For
example, a custom command line argument can be used to control the type of model
that is fit, or how or from where the data is loaded.

First, the :py:func:`build_model` function is where the model that we will fit
will be constructed. The specific model that you choose to construct depends on
your data and your scientific question.

We have to specify a dictionary of model parameter specifications (see
:doc:`models`). Each specification is a dictionary that describes a single
parameter. We can build the model by adjusting predefined sets of model
parameter specifications, stored in the
:py:class:`models.templates.TemplateLibrary` dictionary-like object. In this
example we choose the ``"parametric_sfh"`` set, which has the parameters
necessary for a vasic delay-tau SFH fit with simple attenuation by a dust
screen. This parameter set can be inspected in any of the following ways

.. code-block:: python

		from prospect.models.templates import TemplateLibrary, describe
		# Show basic description of all pre-defined parameter sets
		TemplateLibrary.show_contents()
		# method 1: print the whole dictionary of dictionaries
		model_params = TemplateLibrary["parametric_sfh"]
		print(model_params)
		# Method 2: show a prettier summary of the free and fixed parameters
		print(describe(model_params))

You'll see that this model has 5 free parameters. Any parameter with ``"isfree":
True`` in its specification will be varied during the fit. We have set priors on
these parameters, visible as e.g. ``model_params["mass"]["prior"]``. You may
wish to change the default priors for your particular science case, using the
prior objects in the :py:mod:`models.priors` module. An example of adjusting the
priors for several parameters is given in the :py:func:`build_model` method in
``demo_params.py``. Any free parameter *must* have an associated prior. Other
parameters have their value set to the value of the ``"init"`` key, but do not
vary during the fit. They can be made to vary by setting ``"isfree": True`` and
specifying a prior. Parameters not listed here will be set to their default
values. Typically this means default values in the
:py:class:`fsps.StellarPopulation` object; see `python-fsps
<http://dan.iel.fm/python-fsps/current/>`_ for details. Once you get a set of
parameters from the :py:class:`TemplateLibrary` you can modify or add parameter
specifications. Since ``model_params`` is a dictionary (of dictionaries), you
can update it with other parameter set dictionaries from the
:py:class:`TemplateLibrary`.

Finally, the :py:func:`build_model` function takes the ``model_params``
dictionary that you build and uses it to instantiate a :py:class:`SedModel`
object.

.. code-block:: python

		from prospect.models import SpecModel
		model_params = TemplateLibrary["parametric_sfh"]
		# Turn on nebular emission and add associated parameters
		model_params.update(TemplateLibrary["nebular"])
		model_params["gas_logu"]["isfree"] = True
		model = SpecModel(model_params)
		print(model)


If you wanted to change the specification of the model using custom command line
arguments, you could do it in :py:func:`build_model` by allowing this function
to take keyword arguments with the same name as the custom command line
argument. This can be useful for example to set the initial value of the
redshift ``"zred"`` on an object-by-object basis. Such an example is shown in
``demo_params.py``, which also uses command line arguments to control whether
nebular and/or dust emission parameters are added to the model.


**Building the fit ingredients: build_obs**

The next thing to look at is the :py:func:`build_obs` function. This is where
you take the data from whatever format you have and put it into the format
required by |Codename| for a single object. This means you will have to modify
this function heavily for your own use. But it also means you can use your
existing data formats.

Right now, the :py:func:`build_obs` function just reads ascii data from a file,
picks out a row (corresponding to the photometry of a single galaxy), and then
makes a set of :py:class:`Observation`s using data in that row. You'll note that both the datafile
name and the object number are keyword arguments to this function. That means
they can be set at execution time on the command line, by also including those
variables in the configuration dictionary. We'll see an example later.

When you write your own :py:func:`build_obs` function, you can add all sorts of
keyword arguments that control its output (for example, an object name or ID
number that can be used to choose or find a single object in your data file).
You can also import helper functions and modules. These can be either things
like astropy, h5py, and sqlite or your own project specific modules and
functions. As long as the output data is in the right format (see
dataformat.rst), the body of this function can do anything.

**Building the fit ingredients: the rest**

Ok, now we go to the :py:func:`build_sps` function. This one is pretty
straightforward, it simply instantiates our :py:class:`sources.CSPSpecBasis`
object. For nonparameteric fits one would use the
:py:class:`sources.FastStepBasis` object. These objects hold all the spectral
libraries and produce an SED given a set of parameters. After that is
:py:func:`build_noise`, which is for complexifying the noise model -- ignore
that for now.


Running a fit
----------------------
There are two kinds of fitting packages that can be used with |Codename|. The
first is ``emcee`` which implements ensemble MCMC sampling, and the second is
``dynesty``, which implements dynamic nested sampling. It is also possible to
perform optimization. If ``emcee`` is used, the result of the optimization will
be used to initalize the ensemble of walkers.

The choice of which fitting algorithms to use is based on command line flags
(``--optimization``, ``--emcee``, and ``--dynesty``.) If no flags are set the
model and data objects will be generated and stored in the output file, but no
fitting will take place. To run the fit on object number 0 using ``emcee`` after
an initial optimization, we would do the following at the command line

.. code-block:: shell

		python demo_params.py --objid=0 --emcee --optimize \
		--outfile=demo_obj0_emcee

If we wanted to change something about the MCMC parameters, or fit a different object,
we could also do that at the command line

.. code-block:: shell

		python demo_params.py --objid=1 --emcee --optimize \
		--outfile=demo_obj1_emcee --nwalkers=32 --niter=1024

And if we want to use nested sampling with ``dynesty`` we would do the following

.. code-block:: shell

		python demo_params.py --objid=0  --dynesty \
		--outfile=demo_obj0_dynesty

Finally, it is sometimes useful to run the script from the interpreter to do
some checks. This is best done with the IPython enhanced interactive python.

.. code-block:: shell

		ipython
		In [1]: %run demo_params.py --objid=0 --debug=True

You can then inspect the ``obsdat`` dictionary, the ``model`` object, and the
``run_params`` dictionary to make sure everything is working fine.

To see the full list of available command-line options, you can run the following

.. code-block:: shell

		python demo_params.py --help


Working with the output
--------------------------------
After the fit is completed we should have a file with a name like
``demo_obj0_<fitter>_<timestamp>_mcmc.h5``. This is an HDF5 file containing
sampling results and various configuration data, as well as the observational
data that was fit. By setting ``run_params["output_pickles"]=True`` you can also
output versions of this information in the less portable pickle format. We will
read the HDF5 with python and make some plots using utilities in |Codename|

To read the data back in from the output files that we've generated, use
methods in ``prospect.io.read_results``.

.. code-block:: python

		import prospect.io.read_results as reader
		res, obs, model = reader.results_from("demo_obj_<fitter>_<timestamp>_mcmc.h5")

The ``res`` object is a dictionary containing various useful results. You can
look at ``res.keys()`` to see a list of what it contains. The ``obs`` object is
just the ``obs`` dictionary that was used in the fitting. The ``model`` object
is the model object that was used in the fitting.

**Diagnostic plots**

There are also some methods in this module for basic diagnostic plots. The
``subcorner`` method requires that you have the `corner
<http://corner.readthedocs.io/en/latest/>`_ package installed. It's possible now
to examine the traces (i.e. the evolution of parameter value with MC iteration)
and the posterior PDFs for the parameters.

.. code-block:: python

		# Trace plots
		tfig = reader.traceplot(res)
		# Corner figure of posterior PDFs
		cfig = reader.subcorner(res)


**Working with samples**

If you want to get the *maximum a posteriori* sample, or percentiles of the posterior pdf,
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
        from prospect.plotting.corner import quantile
        weights = res.get("weights", None)
        post_pcts = quantile(flatchain.T, q=[0.16, 0.50, 0.84], weights=weights)


**Stored "best-fit" model**

Further, the prediction of the data for the MAP posterior sample may be stored for you.

.. code-block:: python

        # Plot the stored maximum ln-probability sample
        import matplotlib.pyplot as pl

        best = res["bestfit"]
        a = model.params["zred"] + 1
        pl.plot(a * best["restframe_wavelengths"], best['spectrum'], label="MAP spectrum")
        if obs['filters'] is not None:
            pwave = [f.wave_effective for f in obs["filters"]]
            pl.plot(pwave, best['photometry'], label="MAP photometry")
            pl.set_title(best["parameter"])


This stored best-fit information is only available if the `sps` object was
passed to the :py:func:`write_hdf5` after the fit is run. If it isn't available,
you can regnerate the model predictions for the highest probability sample using
the approach below.

**Regenerating Model predictions**

If necessary, one can regenerate models at any position in the posterior chain.
This requires that we have the sps object used in the fitting to generate
models, which we can regenerate using the :py:func:`read_results.get_sps`
method.

.. code-block:: python

        # We need the correct sps object to generate models
        sps = reader.get_sps(res)


Now we will choose a specific parameter value from the chain and plot what the
observations and the model look like, as well as the uncertainty normalized
residual.  For ``emcee`` results we will use the last iteration of the first
walker, while for ``dynesty`` results we will just use the last sample in the
chain.

.. code-block:: python

        # Choose the walker and iteration number by hand.
        walker, iteration = 0, -1
        if res["chain"].ndim > 2:
            # if you used emcee for the inference
            theta = res['chain'][walker, iteration, :]
        else:
            # if you used dynesty
            theta = res['chain'][iteration, :]

        # Or get a fair sample from the posterior
        from prospect.plotting.utils import sample_posterior
        theta = sample_posterior(res["chain"], weights=res.get("weights", None), nsample=1)[0,:]

        # Get the modeled spectra and photometry.
        # These have the same shape as the obs['spectrum'] and obs['maggies'] arrays.
        (spec, phot), mfrac = model.predict(theta, obs=res['obs'], sps=sps)
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
