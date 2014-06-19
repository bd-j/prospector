from sedpy import observate, attenuation
import sps_basis, sedmodel, elines
from priors import tophat
from gp import GaussianProcess


def parse_argv(argv, **kwargs):
    return rp

def read_params(parfile):
    """
    Return a list of dictionaries
    """
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



#################
#INITIAL GUESS OF SPECTRAL NORMALIZATION USING PHOTOMETRY
#################
#use f475w for normalization
norm_band = [i for i,f in enumerate(model.obs['filters']) if 'f475w' in f.name][0]
synphot = observate.getSED(model.obs['wavelength'], model.obs['spectrum'], model.obs['filters'])

#factor by which model spectra should be multiplied to give you the observed spectra, using the F475W filter as truth
norm = 10**(-0.4 * (synphot[norm_band]  - model.obs['mags'][norm_band]))

#assume you've got this right to within 5% (and 3 sigma) after marginalized over everything
#  that changes spectral shape within the band (polynomial terms, dust, age, etc)
fudge = (1 + 10 * model.obs['mags_unc'][norm_band]/1.086)
fudge = 3.0
model.theta_desc['spec_norm']['prior_args'] = {'mini':norm/fudge, 'maxi':norm * fudge}

#pivot the polynomial near the filter used for approximate normalization
model.params['pivot_wave'] =  model.obs['filters'][norm_band].wave_effective 
model.params['pivot_wave'] = 4750.
if rp['verbose']:
    print('spectral normalization guess = {0}'.format(norm))

initial_center[model.theta_desc['spec_norm']['i0']] = norm
 

    
    return model, initial_theta, sps
