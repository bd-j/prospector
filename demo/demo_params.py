import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


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
                                   "init": luminosity_distance, "units": "Mpc"}

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
    model = sedmodel.SpecModel(model_params)

    return model

# --------------
# Observational Data
# --------------

# Here we are going to put together some filter names
galex = ['galex_FUV', 'galex_NUV']
spitzer = ['spitzer_irac_ch'+n for n in '1234']
bessell = ['bessell_'+n for n in 'UBVRI']
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']

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


def build_obs(objid=0, phottable='demo_photometry.dat',
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

    from prospect.observation import Photometry, Spectrum

    # Here we will read in an ascii catalog of magnitudes as a numpy structured
    # array
    with open(phottable, 'r') as f:
        # drop the comment hash
        header = f.readline().split()[1:]
    catalog = np.genfromtxt(phottable, comments='#',
                            dtype=np.dtype([(n, float) for n in header]))

    # Find the right row
    ind = catalog['objid'] == float(objid)
    # Here we are dynamically choosing which filters to use based on the object
    # and a flag in the catalog.  Feel free to make this logic more (or less)
    # complicated.
    filternames = filtersets[int(catalog[ind]['filterset'])]
    # And here we loop over the magnitude columns
    mags = [catalog[ind]['mag{}'.format(i)] for i in range(len(filternames))]
    mags = np.array(mags)
    # And since these are absolute mags, we can shift to any distance.
    if luminosity_distance is not None:
        dm = 25 + 5 * np.log10(luminosity_distance)
        mags += dm

    # Build output dictionary.
    obs = {}
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filternames` above.
    maggies = np.squeeze(10**(-mags/2.5))
    pdat = Photometry(filters=filternames, flux=maggies,
                      uncertainty=maggies * 0.07, mask=np.isfinite(maggies))
    # We have no spectral data, but we still want to see the predicted spectrum
    sdat = Spectrum()

    # Add unessential bonus info.  This will be stored in output
    pdat.distance_modulus = dm
    pdat.objid = objid

    # This ensures all required keys are present
    pdat.rectify()

    return [sdat, pdat]

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

def build_noise(observations, **extras):
    # use the defaults
    return observations

# -----------
# Everything
# ------------

def build_all(**kwargs):
    observations = build_obs(**kwargs)
    observations = build_noise(observations, **kwargs)
    model = build_model(**kwargs)
    sps = build_sps(**kwargs)

    return (observations, model, sps)


if __name__ == '__main__':

    # --- Parser with default arguments ---
    parser = prospect_args.get_parser()
    # --- Add custom arguments ---
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--luminosity_distance', type=float, default=1e-5,
                        help=("Luminosity distance in Mpc. Defaults to 10pc "
                              "(for case of absolute mags)"))
    parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")

    # --- Configure ---
    args = parser.parse_args()
    config = vars(args)
    config["param_file"] = __file__

    # --- Get fitting ingredients ---
    obs, model, sps = build_all(**config)
    config["sps_libraries"] = sps.ssp.libraries
    print(model)

    if args.debug:
        sys.exit()

    # --- Set up output ---
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = f"{args.outfile}_{ts}_result.h5"

    #  --- Run the actual fit ---
    output = fit_model(obs, model, sps, **config)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile,
                      config,
                      model,
                      obs,
                      output["sampling"],
                      output["optimization"],
                      sps=sps
                      )

    try:
        hfile.close()
    except(AttributeError):
        pass
