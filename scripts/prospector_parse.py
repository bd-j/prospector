import sys
from copy import deepcopy
import numpy as np

from prospect.utils import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


def build_model(zred=0.0, add_neb=True, **extras):
    """Instantiate and return a ProspectorParams model subclass (SedModel).  In
    this example the model Is a simple parameteric SFH (delay-tau) with a Kriek
    & Conroy attenuation model (fixed slope) and a fixed dust emission
    parameters.  Nebular emission is optionally added.
    
    :param zred: (optional, default: 0.1)
        The redshift of the model
        
    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.

    :returns mod:
        A SedModel instance
    """
    # --- Get a basic delay-tau SFH parameter set. ---
    from prospect.models import SedModel
    from prospect.models.templates import TemplateLibrary
    model_params = TemplateLibrary["parametric_sfh"]

    # --- Augment the basic model ----
    model_params.update(TemplateLibrary["burst_sfh"])
    model_params.update(TemplateLibrary["dust_emission"])
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
    # Switch to Kriek and Conroy 2013 for dust
    model_params["dust_type"] = {'N': 1, 'isfree': False,
                                 'init': 4, 'prior': None}
    model_params["dust_index"] = {'N': 1, 'isfree': False,
                                 'init': 0.0, 'prior': None}

    # --- Set dispersions for emcee ---
    model_params["mass"]["init_disp"] = 1e8
    model_params["mass"]["disp_floor"] = 1e7 

    # --- Set initial values ---
    model_params["zred"]["init"] = zred

    return SedModel(model_params)


def build_sps(zcontinuous=1, **extras):
    """Instantiate and return the Stellar Population Synthesis object.  In this
    case, with the parameteric SFH model, we want to use the CSPSpecBasis.

    :param zcontinuous: (default: 1)
        python-fsps parameter controlling how metallicity interpolation of the
        SSPs is acheived.  A value of `1` is recommended.

    :returns sps:
        An *sps* object.
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=False)
    return sps


def build_obs(filterlist=["sdss_r0"], snr=10,
              add_noise=True, seed=0, **run_params):
    """Build a mock observation
    """
    from sedpy import observate
    filters = observate.load_filters(filterlist)

    mock = {"wavelength": None, "spectrum": None, "filters": filters}
    
    # Build the mock model
    sp = build_sps(**run_params)
    mod = build_model(**run_params)
    spec, phot, x = mod.mean_model(mod.theta, mock, sps=sp)

    # Add to dict with uncertainties
    pnoise_sigma = phot / snr
    mock['maggies'] = phot.copy()
    mock['maggies_unc'] = pnoise_sigma

    # And add noise
    if add_noise:
        if int(seed) > 0:
            np.random.seed(int(seed))
        pnoise = np.random.normal(0, 1, len(phot)) * pnoise_sigma
        mock['maggies'] += pnoise

    # Ancillary info
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = phot.copy()
    mock['mock_params'] = deepcopy(mod.params)
    mock['mock_snr_phot'] = snr    
    mock['phot_wave'] = np.array([f.wave_effective for f in mock['filters']])

    obs = mock

    return obs


def build_noise(**run_params):
    """
    """
    return None, None


def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


def setup_h5(emcee=False, outfile=None, model=None, obs=None, **extras):
    """If fitting with emcee, open an hdf5 file and write model, data, and meta
    parameters to the file.  Emcee can then write incrementally to the open
    file.  If not fitting with emcee, just get a timestampped filename.

    :param emcee: (optional, default: False)
        Boolean switch indicating whether emcee sampling is to be performed.

    :returns hfile:
        If `emcee` is True, this is an open :py:class:`h5py.File` handle.
        Otherwise, it is the timestamped default hdf5 filename
    """
    import os, time

    # Try to set up an HDF5 file and write basic info to it
    outroot = "{0}_{1}".format(outfile, int(time.time()))
    odir = os.path.dirname(os.path.abspath(outroot))
    if (not os.path.exists(odir)):
        halt('Target output directory {} does not exist, please make it.'.format(odir))
    hfilename = '{}_mcmc.h5'.format(outroot)

    if not emcee:
        return hfilename
    else:
        import h5py
        hfile = h5py.File(hfilename, "a")
        print("Writing to file {}".format(hfilename))
        writer.write_h5_header(hfile, run_params, model)
        writer.write_obs_to_h5(hfile, obs)
        return hfile


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--zred', type=float, default=0.1,
                        help="redshift for the model")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model")
    parser.add_argument('--add_noise', action="store_true",
                        help="If set, noise up the mock")
    parser.add_argument('--snr', type=float, default=20,
                        help="S/N ratio for the mock photometry")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__
    
    hfile = setup_h5(model=model, obs=obs, **run_params)
    output = fit_model(obs, model, sps, noise, **run_params)
    
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0], 
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])
 
    try:
        hfile.close()
    except(AttributeError):
        pass
