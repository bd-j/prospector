import sys
import pickle
import numpy as np


def use_old_module_names():
    # Fix module renames here if necessary for reading pickle files
    import bsfh.sedmodel as sedmodel
    import bsfh.gp as gp
    import bsfh.sps_basis as sps_basis
    import bsfh.elines as elines
    import bsfh.priors as priors

    sys.modules['sedmodel'] = sedmodel
    sys.modules['gp'] = gp
    sys.modules['sps_basis'] = sps_basis
    sys.modules['elines'] = elines
    sys.modules['priors'] = priors

    
def read_pickles(sample_file, model_file=None,
                 powell_file=None, inmod=None):
    """
    Read a pickle file with stored model and MCMC chains.
    """
    sample_results = pickle.load( open(sample_file, 'rb'))
    powell_results = None
    model = None
    if model_file:
        mf = pickle.load( open(model_file, 'rb'))
        #mf = load(open(model_file, 'rb'))
        inmod = mf['model']
        powell_results = mf['powell']
    if powell_file:
        powell_results = pickle.load( open(powell_file, 'rb'))

    try:
        model = sample_results['model']
    except (KeyError):
        model = inmod
        #model.theta_desc = sample_results['theta']
        sample_results['model'] = model
        
    return sample_results, powell_results, model


def model_comp(theta, model, sps, photflag=0, inlog=True):
    """
    Generate and return various components of the total model for a
    given set of parameters
    """
    obs, _, _ = obsdict(model.obs, photflag=photflag)
    mask = obs['mask']
    mu = model.mean_model(theta, sps=sps)[photflag][mask]
    spec = obs['spectrum'][mask]
    wave = obs['wavelength'][mask]
    
    if photflag == 0:
        cal = model.calibration()[mask]
        try:
            #model.gp.sigma = obs['unc'][mask]/mu
            s = model.params['gp_jitter']
            a = model.params['gp_amplitude']
            l = model.params['gp_length']
            model.gp.factor(s, a, l, check_finite = False, force=True)
            if inlog:
                mu = np.log(mu)
                delta = model.gp.predict(spec - mu - cal)
            else:
                delta = model.gp.predict(spec - mu*cal)
        except:
            delta = 0
    else:
        mask = np.ones(len(obs['wavelength']), dtype= bool)
        cal = np.ones(len(obs['wavelength']))
        delta = np.zeros(len(obs['wavelength']))
        
    return mu, cal, delta, mask, wave

def obsdict(inobs, photflag):
    """
    Return a dictionary of observational data, generated depending on
    whether you're matching photometry or spectroscopy.
    """
    obs = inobs.copy()
    if photflag == 0:
        outn = 'spectrum'
        marker = None
    elif photflag == 1:
        outn = 'sed'
        marker = 'o'
        obs['wavelength'] = np.array([f.wave_effective for f in obs['filters']])
        obs['spectrum'] = 10**(0-0.4 * obs['mags'])
        obs['unc'] = obs['mags_unc'] * obs['spectrum']
        obs['mask'] = obs['mags_unc'] > 0
        
    return obs, outn, marker

def theta_labels(desc):
    """
    Using the theta_desc parameter dictionary, return a list of the model
    parameter names that has the same aorder as the sampling chain array
    """
    label, index = [], []
    for p in desc.keys():
        nt = desc[p]['N']
        name = p
        if p is 'amplitudes':
            name = 'A'
        if nt is 1:
            label.append(name)
            index.append(desc[p]['i0'])
        else:
            for i in xrange(nt):
                label.append(name+'{0}'.format(i+1))
                index.append(desc[p]['i0']+i)

    return [l for (i,l) in sorted(zip(index,label))]
