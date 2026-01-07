import numpy as np
import pytest
from astropy.cosmology import Planck18, WMAP9

from prospect.models import SpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models.transforms import tage_from_tuniv, zred_to_agebins


class DummySPS:
    def get_galaxy_spectrum(self, **kwargs):
        return np.linspace(3000, 8000, 100), np.ones(100), 1.0

    def get_galaxy_elines(self):
        return np.array([3727, 5007]), np.array([1.0, 1.0])

    @property
    def spectral_resolution(self):
        return 0.0


def test_default_cosmology():
    """Test that the default cosmology is Planck18."""
    model_params = TemplateLibrary["ssp"]
    model = SpecModel(model_params)

    # Check if cosmology parameter exists and is Planck18
    assert "cosmology" in model.params
    # Unwrapping checks
    cosmo = model.params["cosmology"]
    if isinstance(cosmo, np.ndarray):
        cosmo = cosmo.item()
    assert cosmo.name == Planck18.name

    # Verify flux_norm uses this cosmology
    # We can check luminosity distance indirectly or spy on it, but checking the object is strong indication

    # Let's check calculations
    z = 1.0
    tage_p18 = tage_from_tuniv(zred=z, tage_tuniv=1.0, cosmology=Planck18)
    tage_default = tage_from_tuniv(
        zred=z, tage_tuniv=1.0, cosmology=None
    )  # Should use default which we set to Planck18 in constants

    # In transforms.py, if cosmology is None, it imports from constants.py
    # In constants.py, we set default to Planck18
    assert np.isclose(tage_p18, tage_default)


def test_custom_cosmology():
    """Test using a custom cosmology (WMAP9)."""
    model_params = TemplateLibrary["ssp"]
    model_params["cosmology"] = {
        "N": 1,
        "isfree": False,
        "init": WMAP9,
        "units": "Astropy Cosmology",
    }
    model = SpecModel(model_params)

    cosmo = model.params["cosmology"]
    if isinstance(cosmo, np.ndarray):
        cosmo = cosmo.item()
    assert cosmo.name == WMAP9.name

    # Verify calculations differ from Planck18
    z = 1.0
    tage_wmap9 = tage_from_tuniv(zred=z, tage_tuniv=1.0, cosmology=WMAP9)
    tage_p18 = tage_from_tuniv(zred=z, tage_tuniv=1.0, cosmology=Planck18)

    assert not np.isclose(tage_wmap9, tage_p18)

    # Verify model uses WMAP9 logic in flux_norm
    model.params["zred"] = np.array([1.0])
    model._zred = model.params["zred"][
        0
    ]  # Manually set _zred as it's usually done in predict_init

    # SpecModel.flux_norm calls cosmo.luminosity_distance(z)
    norm = model.flux_norm()

    # Manually calculate expected norm with WMAP9
    lumdist = WMAP9.luminosity_distance(1.0).to("Mpc").value
    dfactor = (lumdist * 1e5) ** 2
    to_cgs_at_10pc = 3.846e33 / (4.0 * np.pi * (3.085677581467192e18 * 10) ** 2)
    jansky_cgs = 1e-23
    unit_conversion = to_cgs_at_10pc / (3631 * jansky_cgs) * (1 + 1.0)
    expected_norm = 1e10 * unit_conversion / dfactor  # mass is 1e10 by default

    assert np.isclose(norm, expected_norm)


def test_unwrapping_transforms():
    """Test that transforms handle wrapped cosmology objects."""
    z = 0.5

    # Wrapped cosmology
    wrapped_cosmo = np.array([Planck18], dtype=object)

    # Test tage_from_tuniv
    tage_unwrapped = tage_from_tuniv(zred=z, tage_tuniv=1.0, cosmology=Planck18)
    tage_wrapped = tage_from_tuniv(zred=z, tage_tuniv=1.0, cosmology=wrapped_cosmo)
    assert tage_unwrapped == tage_wrapped

    # Test zred_to_agebins
    # Provide multiple bins to avoid index error in zred_to_agebins logic
    agebins = np.array([[0, 8], [8, 9], [9, 10]])
    bins_unwrapped = zred_to_agebins(zred=z, agebins=agebins, cosmology=Planck18)
    bins_wrapped = zred_to_agebins(zred=z, agebins=agebins, cosmology=wrapped_cosmo)
    np.testing.assert_array_equal(bins_unwrapped, bins_wrapped)


def test_sedmodel_unwrapping():
    """Test that SpecModel unwraps cosmology correctly."""
    model_params = TemplateLibrary["ssp"]

    model = SpecModel(model_params)
    model.params["cosmology"] = np.array([Planck18], dtype=object)
    model.params["zred"] = np.array([0.1])
    model._zred = model.params["zred"][0]

    # This should not raise an error
    try:
        model.flux_norm()
    except AttributeError as e:
        pytest.fail(f"SpecModel.flux_norm raised AttributeError: {e}")
    except Exception as e:
        pytest.fail(f"SpecModel.flux_norm raised unexpected exception: {e}")


if __name__ == "__main__":
    test_default_cosmology()
    test_custom_cosmology()
    test_unwrapping_transforms()
    test_sedmodel_unwrapping()
