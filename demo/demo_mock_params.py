import time, sys
from copy import deepcopy
import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format).  See sedpy documentation
galex = ['galex_FUV', 'galex_NUV']
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
spitzer = ['spitzer_irac_ch'+n for n in '1234']


# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose': True,
              'debug': False,
              'outfile': 'output/demo_mock',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee Fitter parameters
              'nwalkers': 64,
              'nburn': [32, 32, 64],
              'niter': 256,
              'interval': 0.25,
              'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi',  # bounding method
              'nested_sample': 'unif',  # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_target_n_effective': 10000,
              # Mock data parameters
              'snr': 20.0,
              'add_noise': False,
              'filterset': galex + sdss + twomass,
              # Input mock model parameters
              'mass': 1e10,
              'logzsol': -0.5,
              'tage': 12.,
              'tau': 3.,
              'dust2': 0.3,
              'zred': 0.1,
              'add_neb': False,
              # SPS parameters
              'zcontinuous': 1,
              }


# --------------
# Model Definition
# --------------

def build_model(zred=0.0, add_neb=True, **extras):
    """Instantiate and return a ProspectorParams model subclass.

    :param zred: (optional, default: 0.0)
        The redshift of the model

    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel
    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parameteric")` to
    # view the parameters, their initial values, and the priors in detail
    model_params = TemplateLibrary["parametric_sfh"]

    # --- Adjust the basic model ----
    # Add burst parameters (fixed to zero be default)
    model_params.update(TemplateLibrary["burst_sfh"])
    # Add dust emission parameters (fixed)
    model_params.update(TemplateLibrary["dust_emission"])
    # Add nebular emission parameters and turn nebular emission on
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])

    # --- Set dispersions for emcee ---
    model_params["mass"]["init_disp"] = 1e8
    model_params["mass"]["disp_floor"] = 1e7

    # --- Complexify dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"] = {'N': 1, 'isfree': False,
                                 'init': 4, 'prior': None}
    # Slope of the attenuation curve, expressed as the index of the power-law
    # that modifies the base Kriek & Conroy/Calzetti shape.
    # I.e. a value of zero is basically calzetti with a 2175AA bump
    model_params["dust_index"] = {'N': 1, 'isfree': False,
                                  'init': 0.0, 'prior': None}

    # --- Set initial values ---
    model_params["zred"]["init"] = zred

    return sedmodel.SedModel(model_params)

# ------------------
# Observational Data
# ------------------

def build_obs(snr=10.0, filterset=["sdss_g0", "sdss_r0"],
              add_noise=True, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.

    :param snr:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock spectrum
    """
    from prospect.utils.obsutils import fix_obs

    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which bands (and wavelengths if doing
    # spectroscopy) in which to generate mock data.
    mock = {}
    mock['wavelength'] = None  # No spectrum
    mock['spectrum'] = None    # No spectrum
    mock['filters'] = load_filters(filterset)

    # We need the models to make a mock
    sps = build_sps(**kwargs)
    mod = build_model(**kwargs)

    # Now we get the mock params from the kwargs dict
    params = {}
    for p in mod.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock
    mod.params.update(params)
    spec, phot, _ = mod.mean_model(mod.theta, mock, sps=sps)

    # Now store some ancillary, helpful info;
    # this information is not required to run a fit.
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = phot.copy()
    mock['mock_params'] = deepcopy(mod.params)
    mock['mock_snr'] = snr
    mock["phot_wave"] = np.array([f.wave_effective for f in mock["filters"]])

    # And store the photometry, adding noise if desired
    pnoise_sigma = phot / snr
    if add_noise:
        pnoise = np.random.normal(0, 1, len(phot)) * pnoise_sigma
        mock['maggies'] = phot + pnoise
    else:
        mock['maggies'] = phot.copy()
    mock['maggies_unc'] = pnoise_sigma
    mock['phot_mask'] = np.ones(len(phot), dtype=bool)

    # This ensures all required keys are present
    mock = fix_obs(mock)

    return mock

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, **extras):
    """Instantiate and return the Stellar Population Synthesis object.

    :param zcontinuous: (default: 1)
        python-fsps parameter controlling how metallicity interpolation of the
        SSPs is acheived.  A value of `1` is recommended.
        * 0: use discrete indices (controlled by parameter "zmet")
        * 1: linearly interpolate in log Z/Z_\sun to the target metallicity
             (the parameter "logzsol".)
        * 2: convolve with a metallicity distribution function at each age.
             The MDF is controlled by the parameter "pmetals"
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=False)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--zred', type=float, default=0.1,
                        help="Redshift for the model (and mock).")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_noise', action="store_true",
                        help="If set, noise up the mock.")
    parser.add_argument('--snr', type=float, default=20,
                        help="S/N ratio for the mock photometry.")
    parser.add_argument('--filterset', type=str, nargs="*",
                        default=galex + sdss + twomass,
                        help="Names of filters through which to produce photometry.")
    parser.add_argument('--tage', type=float, default=12.,
                        help="Age of the mock, Gyr.")
    parser.add_argument('--tau', type=float, default=3.,
                        help="SFH timescale parameter of the mock, Gyr.")
    parser.add_argument('--dust2', type=float, default=0.3,
                        help="Dust attenuation V band optical depth")
    parser.add_argument('--logzsol', type=float, default=-0.5,
                        help="Metallicity of the mock; log(Z/Z_sun)")
    parser.add_argument('--mass', type=float, default=1e10,
                        help="Stellar mass of the mock; solar masses formed")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
