import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from prospect.likelihood.noise_model import NoiseModel_photsamples
from prospect.likelihood.kernels_photsamples import *


# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose': True,
              'debug': False,
              'outfile': 'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee fitting parameters
              'nwalkers': 128,
              'nburn': [16, 32, 64],
              'niter': 512,
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
              'nested_stop_kwargs': {"post_thresh": 0.1},
              # Obs data parameters
              'objid': 0,
              'phottable': 'demo_photometry.dat',
              'luminosity_distance': 1e-5,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# Model Definition
# --------------

def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
                add_neb=False, luminosity_distance=0.0, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # Add lumdist parameter.  If this is not added then the distance is
    # controlled by the "zred" parameter and a WMAP9 cosmology.
    if luminosity_distance > 0:
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units":"Mpc"}

    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 13.
    model_params["mass"]["init"] = 1e8

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["mass"]["init_disp"] = 1e7
    model_params["tau"]["init_disp"] = 3.0
    model_params["tage"]["init_disp"] = 5.0
    model_params["tage"]["disp_floor"] = 2.0
    model_params["dust2"]["disp_floor"] = 0.1

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model

# --------------
# Observational Data
# --------------

def read_fpho_samples(patch_idx=0, convergence_test=True):
    '''
    source activate fpho

    NOTE: currently only reading the first patch
    '''

    import json
    from astropy.io import fits

    base_root = '/home/will/research/phot_covariance/forcepho_output/'
    if convergence_test:
        root = base_root + 'forcepho_convergence_test'
    else:
        root = base_root + 'forcepho_udf_output_example/test_run'
    outname = "outscene_hlf2_xdf"
    #os.chdir(root)
    with open(f"{root}/{outname}_log.json") as f:
        logs = json.load(f)
        slog = logs["sourcelog"]
        plog = logs["patchlog"]

    from forcepho.reconstruction import Samples
    i = plog[patch_idx]
    samples = Samples(f"{root}/patches/patch{i}_samples.h5")

    return samples


def build_obs(objid=0, **kwargs):
    """Load photometry from a a forcepho output.

    :param objid:
        The index of the source to read from the forcepho output. Integer.

    :returns obs:
        Dictionary of observational data.
    """
    from prospect.utils.obsutils import fix_obs
    import sedpy 

    # Load the forcepho photometry samples
    samples = read_fpho_samples()

    # Identify the filters and build the 2d numpy array
    filternames = []
    fmt_acs = 'acs_wfc_%s'
    fmt_wfc3 = 'wfc3_ir_%s'
    phot_samples = np.zeros(( samples.chain.shape[0], samples.bandlist.shape[0] ))
    for i,fn in enumerate(samples.bandlist):
        wave = int(fn.decode()[1:4])
        if wave > 200:
            filternames.append( fmt_acs % fn.decode().lower() )
        else:
            filternames.append( fmt_wfc3 % fn.decode().lower() )
        phot_samples[:,i] = samples.chaincat[ fn.decode() ][objid]

    # Unit conversion: forcepho in nJy --> want maggies
    conv_factor = 1e-9 / 3631  # maggies per nJy
    phot_samples = conv_factor * np.array(phot_samples)

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is the full photometry posterior chain in maggies
    obs['phot_samples'] = phot_samples
    # This is the mean photometry from the chains.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.mean(phot_samples,axis=0)
    obs['maggies_unc'] = np.std(phot_samples,axis=0)
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(obs['maggies']))
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None

    # Add unessential bonus info.  This will be stored in output
    #obs['dmod'] = catalog[ind]['dmod']
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    """build the NoiseModel for the photometry
    return spec_noise, phot_noise
    """

    # kernels:
#    kernel = Uncorrelated_photsamples()
#    kernel = Correlated_photsamples()
    kernel = KDE_photsamples()
    kernel.update(**extras) # <-- NOT CURRENTLY DOING ANYTHING
    phot_noise = NoiseModel_photsamples(metric_name='phot_samples', kernel=kernel)   

    return None, phot_noise


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
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--luminosity_distance', type=float, default=1e-5,
                        help=("Luminosity distance in Mpc. Defaults to 10pc "
                              "(for case of absolute mags)"))
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")

    args = parser.parse_args()
    run_params = vars(args)
    run_params["emcee"] = True
    print(run_params)

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
