import sys, os
import numpy as np

try:
    from sedpy.observate import load_filters
except:
    pass

"""Convenience functions for reading and reconstructing results from a fitting
run, including reconstruction of the model for making posterior samples
"""

__all__ = ["subtriangle", "param_evol"]


def model_comp(theta, model, obs, sps, photflag=0, gp=None):
    """Generate and return various components of the total model for a given
    set of parameters.
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
            gp.kernel[:] = np.log(np.array([s[0], a[0]**2, l[0]**2]))
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
        mask = np.ones(len(obs['wavelength']), dtype=bool)
        cal = np.ones(len(obs['wavelength']))
        delta = np.zeros(len(obs['wavelength']))

    return sed, cal, delta, mask, wave


def param_evol(sample_results, showpars=None, start=0, figsize=None, chains=None, **plot_kwargs):
    """Plot the evolution of each parameter value with iteration #, for each
    walker in the chain.

    :param sample_results:
        A Prospector results dictionary, usually the output of
        ``results_from('resultfile')``.

    :param showpars: (optional)
        A list of strings of the parameters to show.  Defaults to all
        parameters in the ``"theta_labels"`` key of the ``sample_results``
        dictionary.

    :param start: (optional, default: 0)
        Integer giving the iteration number from which to start plotting.

    :param **plot_kwargs:
        Extra keywords are passed to the
        ``matplotlib.axes._subplots.AxesSubplot.plot()`` method.

    :returns tracefig:
        A multipaneled Figure object that shows the evolution of walker
        positions in the parameters given by ``showpars``, as well as
        ln(posterior probability)
    """
    import matplotlib.pyplot as pl

    if chains is None:
        chain = sample_results['chain'][:, start:, :]
        lnprob = sample_results['lnprobability'][:, start:]
    else:
        chain = sample_results['chain'][chains, start:, :]
        lnprob = sample_results['lnprobability'][:, start:]
    nwalk = chain.shape[0]
    try:
        parnames = np.array(sample_results['theta_labels'])
    except(KeyError):
        parnames = np.array(sample_results['model'].theta_labels())

    # logify mass
    if 'mass' in parnames:
        midx = [l=='mass' for l in parnames]
        chain[:,:,midx] = np.log10(chain[:,:,midx])
        parnames[midx] = 'logmass'

    # Restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype=bool)
        parnames = parnames[ind_show]
        chain = chain[:, :, ind_show]

    # Set up plot windows
    ndim = len(parnames) + 1
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim * 1.0 / nx))
    sz = np.array([nx, ny])
    factor = 3.0           # size of one side of one panel
    lbdim = 0.2 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05 * factor         # w/hspace size
    plotdim = factor * sz + factor * (sz - 1) * whspace
    dim = lbdim + plotdim + trdim

    if figsize is None:
        fig, axes = pl.subplots(nx, ny, figsize=(dim[1], dim[0]))
    else:
        fig, axes = pl.subplots(nx, ny, figsize=figsize)
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb[1], bottom=lb[0], right=tr[1], top=tr[0],
                        wspace=whspace, hspace=whspace)

    # Sequentially plot the chains in each parameter
    for i in range(ndim - 1):
        ax = axes.flatten()[i]
        for j in range(nwalk):
            ax.plot(chain[j, :, i], **plot_kwargs)
        ax.set_title(parnames[i], y=1.02)
    # Plot lnprob
    ax = axes.flatten()[-1]
    for j in range(nwalk):
        ax.plot(lnprob[j, :])
    ax.set_title('lnP', y=1.02)
    pl.tight_layout()
    return fig


def subtriangle(sample_results, outname=None, showpars=None,
                start=0, thin=1, truths=None, trim_outliers=None,
                extents=None, **kwargs):
    """Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset of
    the parameters.

    :param start:
        The iteration number to start with when drawing samples to plot.

    :param thin:
        The thinning of each chain to perform when drawing samples to plot.

    :param showpars:
        List of string names of parameters to include in the corner plot.

    :param truths:
        List of truth values for the chosen parameters
    """
    try:
        import triangle
    except(ImportError):
        import corner as triangle

    # pull out the parameter names and flatten the thinned chains
    try:
        parnames = np.array(sample_results['theta_labels'])
    except(KeyError):
        parnames = np.array(sample_results['model'].theta_labels())
    flatchain = sample_results['chain'][:, start::thin, :]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])

    # logify mass
    if 'mass' in parnames:
        midx = [l=='mass' for l in parnames]
        flatchain[:,midx] = np.log10(flatchain[:,midx])
        parnames[midx] = 'logmass'

    # restrict to parameters you want to show
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype=bool)
        flatchain = flatchain[:, ind_show]
        #truths = truths[ind_show]
        parnames = parnames[ind_show]
    if trim_outliers is not None:
        trim_outliers = len(parnames) * [trim_outliers]
    try:
        fig = triangle.corner(flatchain, labels=parnames, truths=truths,  verbose=False,
                              quantiles=[0.16, 0.5, 0.84], range=trim_outliers, **kwargs)
    except:
        fig = triangle.corner(flatchain, labels=parnames, truths=truths,  verbose=False,
                              quantiles=[0.16, 0.5, 0.84], range=trim_outliers, **kwargs)

    if outname is not None:
        fig.savefig('{0}.triangle.png'.format(outname))
        #pl.close(fig)
    else:
        return fig


def obsdict(inobs, photflag):
    """Return a dictionary of observational data, generated depending on
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
