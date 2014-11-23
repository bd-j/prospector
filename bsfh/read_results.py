import sys
import pickle
import numpy as np

"""Convenience functions for reading and reconstruction results from a
fitting run, including reconstruction of the model for posterior
samples"""


def use_old_module_names():
    """Fix module renames here if necessary for reading pickle files
    """
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
                 inmod=None):
    """
    Read a pickle file with stored model and MCMC chains.

    :returns sample_results:
        A dictionary of various results including the sampling chain.

    :returns powell_results:
        A list of the optimizer results for each of the starting conditions.

    :returns model:
        The bsfh.sedmodel object. 
    """
    sample_results = pickle.load( open(sample_file, 'rb'))
    powell_results = None
    model = None
    if model_file:
        try:
            mf = pickle.load( open(model_file, 'rb'))
        except(AttributeError):
            mf = load( open(model_file, 'rb'))
       
        inmod = mf['model']
        powell_results = mf['powell']

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

def subtriangle(sample_results, outname=None, showpars=None,
                start=0, thin=1, truths=None):
    """
    Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset
    of the parameters.
    """
    import triangle
    # pull out the parameter names and flatten the thinned chains
    parnames = np.array(sample_results['model'].theta_labels())
    flatchain = sample_results['chain'][:,start::thin,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    if truths is None:
        truths = sample_results['sampling_initial_center']

    # restrict to parameters you want to show
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype= bool)
        flatchain = flatchain[:,ind_show]
        truths = truths[ind_show]
        parnames= parnames[ind_show]
        
    fig = triangle.corner(flatchain, labels = parnames,
                          quantiles=[0.16, 0.5, 0.84], verbose=False,
                          truths = truths)
    if outname is not None:
        fig.savefig('{0}.triangle.png'.format(outname))
        pl.close()
    else:
        return fig



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
        obs['spectrum'] = obs['maggies']
        obs['unc'] = obs['maggies_unc'] 
        obs['mask'] = obs['phot_mask'] > 0
        
    return obs, outn, marker


## All this because scipy changed
# the name of one class, which shouldn't even be a class.

renametable = {
    'Result': 'OptimizeResult',
    }

def mapname(name):
    if name in renametable:
        return renametable[name]
    return name

def mapped_load_global(self):
    module = mapname(self.readline()[:-1])
    name = mapname(self.readline()[:-1])
    klass = self.find_class(module, name)
    self.append(klass)

def load(file):
    unpickler = pickle.Unpickler(file)
    unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
    return unpickler.load()
