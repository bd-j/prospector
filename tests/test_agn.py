import sys
import numpy as np

import pytest

from prospect.sources import AGNSSPBasis
from prospect.models import SpecModel, templates, priors, sedmodel
from prospect.observation import Spectrum, Photometry
from prospect.models.templates import TemplateLibrary
from prospect.models import priors_beta

@pytest.fixture
def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = AGNSSPBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps


def build_model():
    object_redshift = 3.1
    # The AGN model from Wang+24 currently only works with non-param SFH
    # We use the p-beta template here
    model_params = TemplateLibrary["beta_phisfh"]
    model_params['nzsfh']['prior'] = priors_beta.PhiSFH(zred_mini=object_redshift-0.05, zred_maxi=object_redshift+0.05,
                                                        mass_mini=7.0, mass_maxi=12.5,
                                                        z_mini=-1.98, z_maxi=0.19,
                                                        logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                                        logsfr_ratio_tscale=0.3, nbins_sfh=7,
                                                        const_phi=True)
    model_params.update(TemplateLibrary["nebular"])
    model_params.update(TemplateLibrary["dust_emission"])

    # Adding the AGN continuum
    model_params.update(TemplateLibrary["agn_bbb"])
    # Adding the dust attenuation for the AGN continuum
    model_params.update(TemplateLibrary["dust4"])

    return sedmodel.AGNPolySpecModel(model_params)


def build_obs(**kwargs):

    # Using the phot data from Wang+24 
    
    filternames = ['acs_wfc_f435w', 'acs_wfc_f606w', 'acs_wfc_f814w', 'jwst_f090w', 
                   'wfc3_ir_f105w', 'jwst_f115w', 'wfc3_ir_f125w', 'jwst_f150w', 
                   'wfc3_ir_f160w', 'jwst_f200w', 'jwst_f277w', 'jwst_f356w', 
                   'jwst_f410m', 'jwst_f444w', 'jwst_f770w', 'jwst_f1800w', 'wfc3_ir_f140w']
    maggies = np.array([5.42962186e-12, 3.27244783e-12, 2.94850388e-12, 3.25831441e-12,
                        np.nan, 3.08543201e-12, 5.70936573e-12, 2.78279002e-12,
                        4.72152897e-12, 1.65665362e-11, 5.85270006e-11, 5.91879363e-11,
                        7.33207591e-11, 8.29625132e-11, 1.22583310e-10, 1.78573396e-10,
                        7.39633958e-12])
    maggies_unc = np.array([5.42962186e-12, 3.27244783e-12, 2.94850388e-12, 3.25831441e-12,
                           np.nan, 3.08543201e-12, 5.70936573e-12, 2.78279002e-12,
                           4.72152897e-12, 1.65665362e-11, 5.85270006e-11, 5.91879363e-11,
                           7.33207591e-11, 8.29625132e-11, 1.22583310e-10, 1.78573396e-10,
                           7.39633958e-12])

    pdat = Photometry(filters=filternames, flux=maggies,
                      uncertainty=maggies_unc, mask=np.isfinite(maggies))
    # fake spectrum
    wavelength = np.array([6201.41792534, 6245.98642172, 6291.45766441, 6337.85840424, 6385.21650689])
    flux = np.array([ 1.91111080e-12,  8.45435223e-12,  1.06389410e-11, -1.35803660e-12, 1.03164695e-11])
    flux_unc = np.array([4.54267590e-12, 3.50690798e-12, 5.10245331e-12, 4.38106581e-12, 4.32602334e-12])
    sdat = Spectrum(flux=flux, uncertainty=flux_unc, wavelength=wavelength)

    pdat.rectify()
    sdat.rectify()

    return [sdat, pdat]


def test_agn(build_sps, plot=False):

    sps = build_sps

    obs = build_obs()
    model = build_model()

    theta1 = np.array([4.59275352e-02,  3.15396601e-05,  9.94803031e+00,  1.37694178e+00,
                       1.98329446e-01,  3.11304366e+00,  8.14181339e+00,  1.49392048e-01,
                       -9.60717355e-02,  3.05172117e-01,  8.93662269e-02,  1.22570566e-01,
                       1.02259845e+00,  1.29464564e+00,  7.39998901e+01, -1.45716570e+00,
                       3.76874888e+00])

    pred1 = model.predict(theta1, obs, sps)
    (spec1, phot1), mfrac = pred1

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(model._wave*(1+3.1), model._norm_galspec, label='norm_galspec', color='C0')
        plt.plot(model._wave*(1+3.1), model._norm_agnspec, label='norm_agnspec', color='C3')
        plt.plot(model._wave*(1+3.1), model._norm_agnspec_torus, label='norm_torusspec', color='C5')
        plt.plot(model._wave*(1+3.1), model._norm_spec, label='norm_spec', color='C1')


        plt.xscale('log')
        plt.yscale('log')
        plt.legend()

        plt.ylim(1e-14, 1e-10)
        plt.xlim(2e3, 1e6)
        plt.savefig("prospector_agn.png")
