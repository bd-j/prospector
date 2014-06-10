from sedpy import observate, attenuation
import sps_basis, sedmodel, elines
from priors import tophat
from gp import GaussianProcess


def parse_argv(argv, **kwargs):
    return rp

def read_params(parfile):
    return parlist

def initialize_model(rp, obs):
    parlist = read_params(rp['param_file'])
    
    theta_desc, fixed_params  = {}, {}
    initial_theta = []
    count = 0
    for i, par in enumerate(parlist):
        if par['free']:
            par['i0'] = count
            theta_desc[par['name']] = p
            count += par['N']
            #need to deal with iterables here
            initial_theta.append(par['value'])
        else:
            fixed_params[par['name']] = par['value']

    # SED Model
    model = sedmodel.SedModel(theta_desc = theta_desc, **fixed_params)
    model.ndof = len(model.obs['wavelength']) + len(model.obs['mags'])
    model.verbose = rp['verbose']
    model.add_obs(obs)

    #Add Gaussian Process
    mask = model.obs['mask']
    model.gp = GaussianProcess(model.obs['wavelength'][mask], model.obs['unc'][mask])

    #SPS Model
    sps = sps_basis.StellarPopBasis(smooth_velocity = fixed_params['smooth_velocity'])

    return model, initial_theta, sps
