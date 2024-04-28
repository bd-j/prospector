import sys
import numpy as np

import pytest

from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates, priors
from prospect.observation import Spectrum, Photometry


@pytest.fixture
def build_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def build_model(free_dla=True, damping_wing=False):
    model_params = templates.TemplateLibrary["parametric_sfh"]
    model_params["zred"]["isfree"] = True
    model_params.update(templates.TemplateLibrary["nebular"])
    model_params["nebemlineinspec"]["init"] = False
    lya = "Ly-alpha 1215"
    model_params["elines_to_ignore"] = dict(init=lya, isfree=False)

    # scaling igm_factor is something like x_HI
    model_params.update(templates.TemplateLibrary["igm"])
    model_params["igm_factor"]["isfree"] = True

    # Add the dla column density parameter
    model_params["dla_logNh"] = dict(N=1, isfree=free_dla, init=18,
                                     prior=priors.Uniform(mini=18, maxi=23))

    # Add damping wing switch
    model_params["igm_damping"] = dict(N=1, isfree=False, init=damping_wing)


    return SpecModel(model_params)


def build_obs():

    N = 1000
    wave = np.linspace(0.7e4, 4e4, N)

    lw = ["090w", "115w", "150w", "162m", "182m", "200w", "210m"]
    sw = ["250m", "277w", "300m", "335m", "356w", "410m", "444w"]
    fnames = list([f"jwst_f{b}" for b in sw+lw])
    Nf = len(fnames)
    phot = [Photometry(filters=fnames, flux=np.ones(Nf), uncertainty=np.ones(Nf)/10)]
    spec = [Spectrum(wavelength=wave, flux=np.ones(N),
                     uncertainty=np.ones(N) / 10, mask=slice(None))]

    obslist = spec + phot
    [obs.rectify() for obs in obslist]
    return obslist


def test_dla(build_sps, plot=False):
    sps = build_sps

    obs = build_obs()
    model = build_model()

    model.params["zred"] = 13
    model.params["dla_logNh"] = 0
    model.params["tage"] = 0.4
    model.params["tau"] = 0.2
    model.params["dust2"] = 0.0
    theta1 = model.theta.copy()

    pred1 = model.predict(theta1, obs, sps)
    (spec1, phot1), mfrac = pred1

    if plot:
        import matplotlib.pyplot as pl
        #pl.ion()
        fig, ax = pl.subplots()
        ax.plot(obs[0].wavelength/1e4, spec1)
        ax.plot(obs[1].wavelength/1e4, phot1, "o")

    for nh in [21.5, 22, 23]:
        model.params["dla_logNh"] = nh
        theta = model.theta.copy()
        pred = model.predict(theta, obs, sps)
        (spec, phot), mfrac = pred
        assert np.any(phot < phot1)
        assert np.any(spec < spec1)
        if plot:
            ax.plot(obs[0].wavelength/1e4, spec, label=f"log N_h={nh}")

    if plot:
        ax.plot(obs[1].wavelength/1e4, phot, "o", label=f"log N_h={nh}")
        ax.legend()
        ax.set_xlabel(r"$\lambda (\mu{\rm m})$")
        ax.set_ylabel(r"$f_\nu$")
        fig.savefig("prospector_dla.png")


def test_damping(build_sps, plot=False):

    sps = build_sps

    obs = build_obs()
    model = build_model(damping_wing=False)

    model.params["zred"] = 13
    model.params["dla_logNh"] = 0
    model.params["tage"] = 0.4
    model.params["tau"] = 0.2
    model.params["dust2"] = 0.0
    model.params["igm_damping"] = False
    theta1 = model.theta.copy()

    pred1 = model.predict(theta1, obs, sps)
    (spec1, phot1), mfrac = pred1
    model.params["igm_damping"] = True
    pred2 = model.predict(theta1, obs, sps)
    (spec2, phot2), mfrac = pred2
    assert np.any(phot2 < phot1)
    assert np.any(spec2 < spec1)

    if plot:
        import matplotlib.pyplot as pl
        fig, axes = pl.subplots(2, 1, sharex=True)
        ax = axes[0]
        ax.plot(obs[0].wavelength/1e4, spec2/spec1, label="e^{-tau}")
        ax.legend()
        ax = axes[1]
        ax.plot(obs[0].wavelength/1e4, spec1, label="No damping wing")
        ax.plot(obs[0].wavelength/1e4, spec2, label="with Damping wing")
        ax.legend()
        ax.set_xlabel(r"$\lambda (\mu{\rm m})$")
        ax.set_ylabel(r"$f_\nu$")
        fig.savefig("prospector_damping_wing.png")
