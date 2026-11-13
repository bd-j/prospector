#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import h5py
from sedpy.observate import Filter

from prospect.models import SpecModel, templates
from prospect.sources import CSPSpecBasis
from prospect.observation import Photometry, Spectrum
from prospect.fitting import fit_model
from prospect.io.write_results import write_obs_to_h5
from prospect.io.read_results import obs_from_h5, results_from


SDSS = [f"sdss_{b}0" for b in "ugriz"]


@pytest.fixture(scope="module")
def sps():
    return CSPSpecBasis(zcontinuous=1)


def build_model():
    model_params = templates.TemplateLibrary["parametric_sfh"]
    model_params.update(templates.TemplateLibrary["nebular"])
    # add lines in prospector so nebline_photometry sees the filters too
    model_params["nebemlineinspec"]["init"] = False
    model_params["zred"]["init"] = 0.05
    return SpecModel(model_params)


def as_instance(name):
    """A Filter built from arrays, with the same curve as the named file."""
    native = Filter(name)
    return Filter(f"custom_{name}", data=(native.wavelength, native.transmission))


def gaussian_filter(name, center, sigma):
    wave = np.linspace(center - 4 * sigma, center + 4 * sigma, 81)
    return Filter(name, data=(wave, np.exp(-0.5 * ((wave - center) / sigma)**2)))


def photometry(filters):
    n = len(filters)
    return Photometry(filters=filters, flux=np.ones(n), uncertainty=np.ones(n) / 10)


def test_filter_instances_match_files(sps):
    """Filter instances predict the same maggies as the files they copy,
    alone or mixed with filter names."""
    model = build_model()
    by_name = photometry(SDSS)
    by_instance = photometry([as_instance(n) for n in SDSS])
    mixed = photometry(SDSS[:2] + [as_instance(n) for n in SDSS[2:]])

    assert by_instance.filternames == [f"custom_{n}" for n in SDSS]
    assert mixed.filternames == SDSS[:2] + [f"custom_{n}" for n in SDSS[2:]]

    (expected,), _ = model.predict(model.theta, [by_name], sps=sps)
    assert np.all(expected > 0)
    for obs in (by_instance, mixed):
        (pred,), _ = model.predict(model.theta, [obs], sps=sps)
        np.testing.assert_allclose(pred, expected, rtol=1e-12)


def test_filter_curves_roundtrip(tmp_path):
    """Filter instance curves are written with the photometry and rebuilt on
    read, while named filters still load by name."""
    phot = photometry(["sdss_r0", gaussian_filter("gauss", 8000, 100)])
    fn = tmp_path / "obs.h5"
    with h5py.File(fn, "w") as hf:
        write_obs_to_h5(hf, [phot])
    with h5py.File(fn, "r") as hf:
        (reread,) = obs_from_h5(hf["observations"])

    assert reread.filternames == ["sdss_r0", "gauss"]
    np.testing.assert_array_equal(reread.flux, phot.flux)
    np.testing.assert_array_equal(reread.filterset.lam, phot.filterset.lam)
    np.testing.assert_array_equal(reread.filterset.trans, phot.filterset.trans)


def test_named_filters_store_no_curves():
    """Photometry with only filter names serializes as before."""
    phot = photometry(SDSS)
    assert phot.filter_wavelength is None
    assert "filter_wavelength" not in phot.to_struct().dtype.names


def test_empty_spectrum_fit(sps, tmp_path):
    """An empty Spectrum passes rectify, survives an optimization alongside
    custom-filter photometry, and both read back from the results file."""
    model = build_model()
    filters = [gaussian_filter(f"g{i}", w, w / 80)
               for i, w in enumerate(np.linspace(7500, 50000, 12))]
    phot = photometry(filters)
    (phot.flux,), _ = model.predict(model.theta, [phot], sps=sps)
    phot.uncertainty = phot.flux / 20
    spec = Spectrum()
    observations = [phot, spec]

    spec.rectify()
    assert spec.ndof == 0

    run_params = dict(optimize=True, min_method="lm", nmin=1,
                      min_opts=dict(max_nfev=20))
    output = fit_model(observations, model, sps, **run_params)
    assert output["optimization"] is not None

    (pphot, pspec), _ = model.predict(model.theta, observations, sps=sps)
    assert len(pphot) == len(filters)
    assert len(pspec) == len(sps.wavelengths)

    fn = tmp_path / "results.h5"
    with h5py.File(fn, "w") as hf:
        write_obs_to_h5(hf, observations)
    _, reread, _ = results_from(str(fn), dangerous=False)
    assert reread is not None
    reread = {o.kind: o for o in reread}
    assert set(reread) == {"photometry", "spectrum"}
    assert reread["spectrum"].flux is None
    assert reread["photometry"].filternames == [f.name for f in filters]
    np.testing.assert_array_equal(reread["photometry"].filterset.trans,
                                  phot.filterset.trans)


def test_scalar_response_serializes():
    """A Spectrum whose response was set to a scalar still converts to a
    structured array, without a response column."""
    n = 5
    spec = Spectrum(wavelength=np.linspace(4000, 5000, n), flux=np.ones(n),
                    uncertainty=np.ones(n) / 10)
    spec.compute_response(spec=np.ones(n))
    assert np.ndim(spec.response) == 0
    struct = spec.to_struct()
    assert "response" not in struct.dtype.names
    assert len(struct) == n
