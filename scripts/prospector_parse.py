from prospect import argument_parser
from prospect.fitting import fit_model


def build_model(zred=0.0, add_neb=True, **run_params):
    """Instantiate and return a ProspectorParams model subclass.
    
    :param zred: (optional, default: 0.1)
        The redshift of the model
        
    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.

    :returns mod:
        A SedModel instance
    """
    # --- Get a basic delay-tau SFH parameter set. ---
    from prospect.models.templates import TemplateLibrary
    model_params = TemplateLibrary["parametric"]

    # --- Augment the basic model ----
    model_params.update(TemplateLibrary["burst"])
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

    return sedmodel.SedModel(model_params)


def build_sps(zcontinuous=1, **run_params):
    """Instantiate and return the Stellar Population Synthesis object.

    :param zcontinuous: (default: 1)
        python-fsps parameter controlling how metallicity interpolation of the
        SSPs is acheived.  A value of `1` is recommended.

    :returns sps:
        An *sps* object.
    """
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=False)
    return sps


def build_obs(**run_params):
    """
    """
    return obs


def build_noise(**run_params):
    """
    """
    return None None


def build_all(**kwargs):
    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':
    parser = prospector_argparser.get_parser()  # parser with default arguments 
    parser.add_argument('--custom_argument_1', ...)
    parser.add_argument('--custom_argument_2', ...)
    
    args = parser.parse_args()
    run_params = vars(args)

    obs, mod, sps, noise = build_all(**run_params)
    fit_model(obs, mod, sps, noise, **run_params)
