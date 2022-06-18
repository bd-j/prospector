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


Build an observation
--------------------

First we'll get some data, using astroquery to get SDSS photometry of a galaxy.  We'll also
get spectral data so we know the redshift.

.. code:: python

    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    bands = "ugriz"
    mcol = [f"cModelMag_{b}" for b in bands]
    ecol = [f"cModelMagErr_{b}" for b in bands]
    cat = SDSS.query_crossid(SkyCoord(ra=204.46376, dec=35.79883, unit="deg"), data_release=16,
                             photoobj_fields=mcol + ecol + ["specObjID"])
    shdus = SDSS.get_spectra(plate=2101, mjd=53858, fiberID=220)[0]
    assert int(shdus[2].data["SpecObjID"][0]) == cat[0]["specObjID"]

Now we will put this data in the format expected by prospector. We convert the
magnitudes to maggies, convert the magnitude errors to flux uncertainties
(including a noise floor), and load the filter transmission curves using
`sedpy`. We'll store the redshift here as well for convenience.  Note that for
this example we do *not* attempt to fit the spectrum at the same time, though we
include an empty Spectrum data set to force a prediction of the full spectrum.

.. code:: python

    from sedpy.observate import load_filters
    from prospect.data import Photometry, Spectrum

    filters = load_filters([f"sdss_{b}0" for b in bands])
    maggies = np.array([10**(-0.4 * cat[0][f"cModelMag_{b}"]) for b in bands])
    magerr = np.array([cat[0][f"cModelMagErr_{b}"] for b in bands])
    magerr = np.clip(magerr, 0.05, np.inf)

    pdat = Photometry(filters=filters, flux=maggies, uncertainty=magerr*maggies/1.086,
                      name=f'sdss_phot_specobjID{cat[0]["specObjID"]}')
    sdat = Spectrum(wavelength=None, flux=None, uncertainty=None)
    observations = [sdat, pdat]
    for obs in observations:
        obs.redshift = shdus[2].data[0]["z"]

In principle we could also add noise models for the spectral and photometric
data (e.g. to fit for the photometric noise floor), but we'll make the default
assumption of iid Gaussian noise for the moment.


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
    (spec, phot), mfrac = model.predict(model.theta, observations, sps=sps)
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
    fitting_kwargs = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=10000)
    output = fit_model(obs, model, sps, optimize=False, dynesty=True, lnprobfn=lnprobfn, **fitting_kwargs)
    result, duration = output["sampling"]

The result is a dictionary with keys giving the Monte Carlo samples of parameter
values and the corresponding posterior probabilities.  Because we are using
dynesty, we also get weights associated with each parameter sample in the chain.

Typically we'll want to save the fit information.  We can save the output of the
sampling along with other information about the model and the data that was fit
using the following:

.. code:: python

    from prospect.io import write_results as writer
    hfile = "quickstart_dynesty_mcmc.h5"
    writer.write_hdf5(hfile, {}, model, obs,
                     output["sampling"][0], None,
                     sps=sps,
                     tsample=output["sampling"][1],
                     toptimize=0.0)