Quickstart
==========

Here's a quick intro that fits 5-band SDSS photometry with a simple delay-tau
parametric SFH model. This assumes you've successully installed prospector and
all the prerequisites.  This is intended simply to introduce the key
ingrediants; for more realistic usage see :ref:`demo` or the :ref:`tutorial`.

.. code:: python

    import fsps
    import dynesty
    import sedpy
    import h5py, astropy
    import numpy as np
    import astroquery


Build an observation
--------------------

First we'll get some data, using ``astroquery`` to get SDSS photometry of a galaxy.  We'll also
get spectral data so we know the redshift.

.. code:: python

    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    bands = "ugriz"
    mcol = [f"cModelMag_{b}" for b in bands]
    ecol = [f"cModelMagErr_{b}" for b in bands]
    cat = SDSS.query_crossid(SkyCoord(ra=204.46376, dec=35.79883, unit="deg"),
                             data_release=16,
                             photoobj_fields=mcol + ecol + ["specObjID"])
    shdus = SDSS.get_spectra(plate=2101, mjd=53858, fiberID=220)[0]
    assert int(shdus[2].data["SpecObjID"][0]) == cat[0]["specObjID"]

Now we will put this data in a dictionary with format expected by prospector. We
convert the magnitudes to maggies, convert the magnitude errors to flux
uncertainties (including a noise floor), and load the filter transmission curves
using `sedpy`. We'll store the redshift here as well for convenience.  Note that
for this example we do *not* attempt to fit the spectrum at the same time.

.. code:: python

    from sedpy.observate import load_filters
    from prospect.utils.obsutils import fix_obs

    filters = load_filters([f"sdss_{b}0" for b in bands])
    maggies = np.array([10**(-0.4 * cat[0][f"cModelMag_{b}"]) for b in bands])
    magerr = np.array([cat[0][f"cModelMagErr_{b}"] for b in bands])
    magerr = np.clip(magerr, 0.05, np.inf)

    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=shdus[2].data[0]["z"],
               maggies=maggies, maggies_unc=magerr * maggies / 1.086, filters=filters)
    obs = fix_obs(obs)


Build a Model
-------------

Here we will get a default parameter set for a simple parametric SFH, and add a
set of parameters describing nebular emission.  We'll also fix the redshift to
the value given by SDSS. This model has 5 free parameters, each of which has a
an associated prior distribution.  These default prior distributions can and
should be replaced or adjusted depending on your particular science question.

.. code:: python

    from prospect.models.templates import TemplateLibrary
    from prospect.models import SpecModel
    model_params = TemplateLibrary["parametric_sfh"]
    model_params.update(TemplateLibrary["nebular"])
    model_params["zred"]["init"] = obs["redshift"]

    model = SpecModel(model_params)
    assert len(model.free_params) == 5
    print(model)

In principle we could also add noise models for the spectral and photometric
data, but we'll make the default assumption of iid Gaussian noise for the moment.

.. code:: python

    noise_model = (None, None)


Get a 'Source'
--------------

Now we need an object that will actually generate the galaxy spectrum using
stellar population synthesis.  For this we will use an object that wraps FSPS
allowing access to all the parameterized SFHs.  We will also just check which
spectral and isochrone librariews are being used.

.. code:: python

    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=1)
    print(sps.ssp.libraries)


Make a prediction
-----------------

We can now predict our data for any set of parameters.  This will take a little
time because fsps is building and caching the SSPs.  Subsequent calls to predict
will be faster.  Here we'll just make the predicition for the current value of
the free parameters.

.. code:: python

    current_parameters = ",".join([f"{p}={v}" for p, v in zip(model.free_params, model.theta)])
    print(current_parameters)
    spec, phot, mfrac = model.predict(model.theta, obs=obs, sps=sps)
    print(phot / obs["maggies"])


Run a fit
---------

Since we can make predictions and we have data and uncertainties, we should be
able to construct a likelihood function.  Here we'll use the pre-defined default
posterior probability function.  We also set some some sampling related keywords
to make the fit go a little faster, though it should still take of order tens of
minutes.

.. code:: python

    from prospect.fitting import lnprobfn, fit_model
    fitting_kwargs = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=1000)
    output = fit_model(obs, model, sps, optimize=False, dynesty=True, lnprobfn=lnprobfn, noise=noise_model, **fitting_kwargs)
    result, duration = output["sampling"]

The ``result`` is a dictionary with keys giving the Monte Carlo samples of
parameter values and the corresponding posterior probabilities.  Because we are
using dynesty, we also get weights associated with each parameter sample in the
chain.

Typically we'll want to save the fit information.  We can save the output of the
sampling along with other information about the model and the data that was fit
as follows:

.. code:: python

    from prospect.io import write_results as writer
    hfile = "./quickstart_dynesty_mcmc.h5"
    writer.write_hdf5(hfile, {}, model, obs,
                     output["sampling"][0], None,
                     sps=sps,
                     tsample=output["sampling"][1],
                     toptimize=0.0)

Make plots
----------

Now we'll want to read the saved fit information and make plots. To read the
information we can use the built-in reader.

.. code:: python

    from prospect.io import read_results as reader
    hfile = "./quickstart_dynesty_mcmc.h5"
    out, out_obs, out_model = reader.results_from(hfile)

This gives a dictionary of useful information (``out``), as well as the obs
dictionary that we were using and, in some cases, a reconsitituted model object.
However, that is only possible if the model generation code is saved to the
results file, in the form of the text for a `build_model()` function.  Here we
will use just use the model object that we've already generated.

Now we will do some plotting.  First, lets make a corner plot of the posterior.
We'll mark the highest probablity posterior sample as well.

.. code:: python

    import matplotlib.pyplot as pl
    from prospect.plotting import corner
    nsamples, ndim = out["chain"].shape
    cfig, axes = pl.subplots(ndim, ndim, figsize=(10,9))
    axes = corner.allcorner(out["chain"].T, out["theta_labels"], axes, weights=out["weights"], color="royalblue", show_titles=True)

    from prospect.plotting.utils import best_sample
    pbest = best_sample(out)
    corner.scatter(pbest[:, None], axes, color="firebrick", marker="o")

Note that the highest probability sample may well be different than the peak of
the marginalized posterior distribution.

Now let's plot the observed SED and the spectrum and SED of the highest
probability posterior sample.

.. code:: python

    import matplotlib.pyplot as pl
    sfig, saxes = pl.subplots(2, 1, gridspec_kw=dict(height_ratios=[1, 4]), sharex=True)
    ax = saxes[1]
    pwave = np.array([f.wave_effective for f in out_obs["filters"]])
    # plot the data
    ax.plot(pwave, out_obs["maggies"], linestyle="", marker="o", color="k")
    ax.errorbar(pwave,  out_obs["maggies"], out_obs["maggies_unc"], linestyle="", color="k", zorder=10)
    ax.set_ylabel(r"$f_\nu$ (maggies)")
    ax.set_xlabel(r"$\lambda$ (AA)")
    ax.set_xlim(3e3, 1e4)
    ax.set_ylim(out_obs["maggies"].min() * 0.1, out_obs["maggies"].max() * 5)
    ax.set_yscale("log")

    # get the best-fit SED
    bsed = out["bestfit"]
    ax.plot(bsed["restframe_wavelengths"] * (1+out_obs["redshift"]), bsed["spectrum"], color="firebrick", label="MAP sample")
    ax.plot(pwave, bsed["photometry"], linestyle="", marker="s", markersize=10, mec="orange", mew=3, mfc="none")

    ax = saxes[0]
    chi = (out_obs["maggies"] - bsed["photometry"]) / out_obs["maggies_unc"]
    ax.plot(pwave, chi, linestyle="", marker="o", color="k")
    ax.axhline(0, color="k", linestyle=":")
    ax.set_ylim(-2, 2)
    ax.set_ylabel(r"$\chi_{\rm best}$")

Sometimes it is desirable to reconstitute the SED from a particular posterior
sample or set of samples, or even the spectrum of the highest probability sample
if it was not saved.  This requires both the model and the sps object generated
previously.

.. code:: python

    from prospect.plotting.utils import sample_posterior
    # Here we fairly and randomly choose a posterior sample
    p = sample_posterior(out["chain"], weights=out["weights"], nsample=1)
    # show this sample in the corner plot
    corner.scatter(p.T, axes, color="darkslateblue", marker="o")
    # regenerate the spectrum and plot it
    spec, phot, mfrac = model.predict(p[0], obs=out_obs, sps=sps)
    ax = saxes[1]
    ax.plot(sps.wavelengths * (1+out_obs["redshift"]), spec, color="darkslateblue", label="posterior sample")