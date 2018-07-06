import numpy as np
from prospect.models import priors, SedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters

# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol':0.5e-5, 'maxfev': 5000,
              'do_levenburg': True,
              'nmin': 10,
              # emcee fitting parameters
              'nwalkers':128,
              'nburn': [16, 32, 64],
              'niter': 512,
              'interval': 0.25,
              'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'unif', # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_stop_kwargs': {"post_thresh": 0.1},
              # Obs data parameters
              'objid':0,
              'phottable': 'demo_photometry.dat',
              'logify_spectrum':False,
              'normalize_spectrum':False,
              'luminosity_distance': 1e-5,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_dust': False,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# OBS
# --------------

# Here we are going to put together some filter names
galex = ['galex_FUV', 'galex_NUV']
spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]
bessell = ['bessell_'+n for n in ['U', 'B', 'V', 'R', 'I']]
sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]

# The first filter set is Johnson/Cousins, the second is SDSS. We will use a
# flag in the photometry table to tell us which set to use for each object
# (some were not in the SDSS footprint, and therefore have Johnson/Cousins
# photometry)
#
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
filtersets = (galex + bessell + spitzer,
              galex + sdss + spitzer)


def load_obs(objid=0, phottable='demo_photometry.dat',
             luminosity_distance=None, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)

    # Here we will read in an ascii catalog of magnitudes as a numpy structured
    # array
    with open(phottable, 'r') as f:
        # drop the comment hash
        header = f.readline().split()[1:]
    catalog = np.genfromtxt(phottable, comments='#',
                            dtype=np.dtype([(n, np.float) for n in header]))

    # Find the right row
    ind = catalog['objid'] == float(objid)
    # Here we are dynamically choosing which filters to use based on the object
    # and a flag in the catalog.  Feel free to make this logic more (or less)
    # complicated.
    filternames = filtersets[ int(catalog[ind]['filterset']) ]
    # And here we loop over the magnitude columns
    mags = [catalog[ind]['mag{}'.format(i)] for i in range(len(filternames))]
    mags = np.array(mags)
    # And since these are absolute mags, we can shift to any distance.
    if luminosity_distance is not None:
        dm = 25 + 5 * np.log10(luminosity_distance)
        mags += dm

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.squeeze(10**(-mags/2.5))
    # HACK.  You should use real flux uncertainties
    obs['maggies_unc'] = obs['maggies'] * 0.07
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # We have no spectrum.
    obs['wavelength'] = None

    # Add unessential bonus info.  This will be stored in output
    #obs['dmod'] = catalog[ind]['dmod']
    obs['objid'] = objid

    return obs


# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps


# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    return None, None


# --------------
# MODEL_PARAMS
# --------------

def load_model(object_redshift=None, fixed_metallicity=None, add_dust=False,
               add_neb=False, luminosity_distance=None, **extras):
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
    if luminosity_distance is not None:
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units":"Mpc"}

    # Adjust model initial values (only important for emcee)
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

    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_dust:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = SedModel(model_params)

    return model

