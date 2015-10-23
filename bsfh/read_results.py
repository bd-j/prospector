import sys
import pickle
import numpy as np
import matplotlib.pyplot as pl

"""Convenience functions for reading and reconstruction results from a
fitting run, including reconstruction of the model for posterior
samples"""
    
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

def model_comp(theta, model, obs, sps, photflag=0, gp=None):
    """
    Generate and return various components of the total model for a
    given set of parameters
    """
    logarithmic = obs.get('logify_spectrum')
    obs, _, _ = obsdict(obs, photflag=photflag)    
    mask = obs['mask']
    mu = model.mean_model(theta, obs, sps=sps)[photflag][mask]
    sed = model.sed(theta, obs, sps=sps)[photflag][mask]
    wave = obs['wavelength'][mask]
    
    if photflag == 0:
        cal = model.spec_calibration(theta, obs)
        if type(cal) is not float:
            cal = cal[mask]
        try:
            s, a, l = model.spec_gp_params()
            gp.kernel[:] = np.log(np.array([s[0],a[0]**2,l[0]**2]))
            spec = obs['spectrum'][mask]
            if logarithmic:
                gp.compute(wave, obs['unc'][mask])
                delta = gp.predict(spec - mu, wave)
            else:
                gp.compute(wave, obs['unc'][mask], flux=mu)
                delta = gp.predict(spec - mu)
            if len(delta) == 2:
                delta = delta[0]
        except(TypeError, AttributeError, KeyError):
            delta = 0
    else:
        mask = np.ones(len(obs['wavelength']), dtype= bool)
        cal = np.ones(len(obs['wavelength']))
        delta = np.zeros(len(obs['wavelength']))
        
    return sed, cal, delta, mask, wave

def param_evol(sample_results, outname=None, showpars=None, start=0, **plot_kwargs):
    """
    Plot the evolution of each parameter value with iteration #, for
    each chain.
    """
    
    chain = sample_results['chain'][:,start:,:]
    lnprob = sample_results['lnprobability'][:,start:]
    nwalk = chain.shape[0]
    parnames = np.array(sample_results['model'].theta_labels())

    #restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype= bool)
        parnames = parnames[ind_show]
        chain = chain[:,:,ind_show]

    #set up plot windows
    ndim = len(parnames) +1
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim*1.0/nx))
    sz = np.array([nx,ny])
    factor = 3.0           # size of one side of one panel
    lbdim = 0.2 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05*factor         # w/hspace size
    plotdim = factor * sz + factor *(sz-1)* whspace
    dim = lbdim + plotdim + trdim
    
    fig, axes = pl.subplots(nx, ny, figsize = (dim[1], dim[0]))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb[1], bottom=lb[0], right=tr[1], top=tr[0],
                        wspace=whspace, hspace=whspace)

    #sequentially plot the chains in each parameter
    for i in range(ndim-1):
        ax = axes.flatten()[i]
        for j in range(nwalk):
            ax.plot(chain[j,:,i], **plot_kwargs)
        ax.set_title(parnames[i])
    #plot lnprob
    ax = axes.flatten()[-1]
    for j in range(nwalk):
        ax.plot(lnprob[j,:])
    ax.set_title('lnP')
    if outname is not None:
        fig.savefig('{0}.x_vs_step.png'.format(outname))
        pl.close()
    else:
        return fig


def subtriangle(sample_results, outname=None, showpars=None,
                start=0, thin=1, truths=None):
    """
    Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset
    of the parameters.

    :param start:
        The iteration number to start with when drawing samples to plot.

    :param thin:
        The thinning of each chain to perform when drawing samples to plot.

    :param showpars:
        List of string names of parameters to include in the corner plot.

    :param truths:
        List of truth values
    """
    import triangle
    # pull out the parameter names and flatten the thinned chains
    parnames = np.array(sample_results['model'].theta_labels())
    flatchain = sample_results['chain'][:,start::thin,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    #if truths is None:
    #    truths = sample_results['sampling_initial_center']

    # restrict to parameters you want to show
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype= bool)
        flatchain = flatchain[:,ind_show]
        #truths = truths[ind_show]
        parnames= parnames[ind_show]
        
    fig = triangle.corner(flatchain, labels = parnames,
                          quantiles=[0.16, 0.5, 0.84], verbose=False,
                          truths = truths)
    if outname is not None:
        fig.savefig('{0}.triangle.png'.format(outname))
        pl.close(fig)
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

# All this because scipy changed the name of one class, which
# shouldn't even be a class.

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
