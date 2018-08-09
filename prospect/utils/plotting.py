import numpy as np
import matplotlib.pyplot as pl

__all__ = ["get_best", "get_truths", "get_percentiles", "get_stats",
           "posterior_samples", "hist_samples", "joint_pdf", "compute_sigma_level",
           "get_prior", "trim_walkers", "fill_between", "figgrid"]


def get_best(res, **kwargs):
    """Get the maximum a posteriori parameters.
    """
    imax = np.argmax(res['lnprobability'])
    # there must be a more elegant way to deal with differnt shapes
    try:
        i, j = np.unravel_index(imax, res['lnprobability'].shape)
        theta_best = res['chain'][i, j, :].copy()
    except(ValueError):
        theta_best = res['chain'][imax, :].copy()

    theta_names = res.get('theta_labels', res['model'].theta_labels())
    return theta_names, theta_best


def get_truths(res):
    import pickle
    try:
        mock = pickle.loads(res['obs']['mock_params'])
        res['obs']['mock_params'] = mock
    except:
        pass
    try:
        return res['obs']['mock_params']
    except(KeyError):
        return None


def get_percentiles(res, ptile=[16, 50, 84], start=0.0, thin=1, **extras):
    """Get get percentiles of the marginalized posterior for each parameter.

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param ptile: (optional, default: [16, 50, 84])
       A list of percentiles (integers 0 to 100) to return for each parameter.

    :param start: (optional, default: 0.5)
       How much of the beginning of chains to throw away before calculating
       percentiles, expressed as a fraction of the total number of iterations.

    :param thin: (optional, default: 10.0)
       Only use every ``thin`` iteration when calculating percentiles.

    :returns pcts:
       Dictionary with keys giving the parameter names and values giving the
       requested percentiles for that parameter.
    """

    parnames = np.array(res.get('theta_labels', res['model'].theta_labels()))
    niter = res['chain'].shape[-2]
    start_index = np.floor(start * (niter-1)).astype(int)
    if res["chain"].ndim > 2:
        flatchain = res['chain'][:, start_index::thin, :]
        dims = flatchain.shape
        flatchain = flatchain.reshape(dims[0]*dims[1], dims[2])
    elif res["chain"].ndim == 2:
        flatchain = res["chain"][start_index::thin, :]
    pct = np.array([quantile(p, ptile, weights=res.get("weights", None)) for p in flatchain.T])
    return dict(zip(parnames, pct))


def quantile(data, percents, weights=None):
    ''' percents in units of 1%
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1.*w.cumsum()/w.sum()*100
    y = np.interp(percents, p, d)
    return y


def get_stats(res, pnames, **kwargs):
    """For selected parameters, get the truth (if known), the MAP value from
    the chain, and the percentiles.

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param pnames:
        List of strings giving the names of the desired parameters.
    """
    truth = get_truths(res)
    best = dict(zip(*get_best(res)))
    pct = get_percentiles(res, **kwargs)

    if truth is not None:
        truths = np.array([truth[k] for k in pnames])
    else:
        truths = None
    pcts = np.array([pct[k] for i,k in enumerate(pnames)])
    bests = np.array([best[k] for i,k in enumerate(pnames)])

    return pnames, truths, bests, pcts


def trim_walkers(res, threshold=-1e4):
    """Remove walkers with probability below some threshold.  Useful for
    removing stuck walkers
    """
    good = res['lnprobability'][:, -1] > threshold
    trimmed = {}
    trimmed['chain'] = res['chain'][good, :, :]
    trimmed['lnprobability'] = res['lnprobability'][good, :]
    trimmed['model'] = res['model']
    return trimmed


def joint_pdf(res, p1, p2, pmap={}, **kwargs):
    """Build a 2-dimensional array representing the binned joint PDF of 2
    parameters, in terms of sigma or fraction of the total distribution.

    For example, to plot contours of the joint PDF of parameters ``"parname1"``
    and ``"parname2"`` from the last half of a chain with 30bins in each
    dimension;

    ::

        xb, yb, sigma = joint_pdf(res, parname1, parname2, nbins=30, start=0.5)
        ax.contour(xb, yb, sigma, **plotting_kwargs)

    :param p1:
        The name of the parameter for the x-axis

    :param p2:
        The name of the parameter for the y axis

    :returns xb, yb, sigma:
        The bins and the 2-d histogram
    """
    trace, pars = hist_samples(res, [p1, p2], **kwargs)
    trace = trace.copy().T
    if pars[0] == p1:
        trace = trace[::-1, :]
    x = pmap.get(p2, lambda x: x)(trace[0])
    y = pmap.get(p1, lambda x: x)(trace[1])
    xbins, ybins, sigma = compute_sigma_level(x, y, **kwargs)
    return xbins, ybins, sigma.T


def posterior_samples(res, samples=[1.0], **kwargs):
    """Pull samples of theta from the MCMC chain

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param samples:
        Iterable of random numbers between 0 and 1.

    :param **kwargs:
        Extra keywords are passed to ``hist_samples``.

    :returns thetas:
        A list of parameter vectors pulled at random from the chain, of same
        length as ``samples``.
    """
    flatchain, pnames = hist_samples(res, **kwargs)
    ns = flatchain.shape[0]
    thetas = [flatchain[s, :]
              for s in np.floor(np.array(samples) * (ns-1)).astype(int)]
    return thetas


def hist_samples(res, showpars=None, start=0, thin=1,
                 return_lnprob=False, **extras):
    """Get posterior samples for the parameters listed in ``showpars``.  This
    can be done for different ending fractions of the (thinned) chain.

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param showpars:
        A list of strings giving the desired parameters.

    :param start: (optional, default: 0.5)
       How much of the beginning of chains to throw away before calculating
       percentiles, expressed as a fraction of the total number of iterations.

    :param thin: (optional, default: 10.0)
       Only use every ``thin`` iteration when calculating percentiles.
    """
    parnames = np.array(res.get('theta_labels', res['model'].theta_labels()))
    niter = res['chain'].shape[-2]
    start_index = np.floor(start * (niter-1)).astype(int)
    if res["chain"].ndim > 2:        
        flatchain = res['chain'][:, start_index::thin, :]
        dims = flatchain.shape
        flatchain = flatchain.reshape(dims[0]*dims[1], dims[2])
    elif res["chain"].ndim == 2:
        flatchain = res["chain"][start_index::thin, :]
    if showpars is None:
        ind_show = slice(None)
    else:
        ind_show = np.array([p in showpars for p in parnames], dtype= bool)
    flatchain = flatchain[:, ind_show]
    if return_lnprob:
        flatlnprob = res['lnprobability'][:, start_index::thin].reshape(dims[0]*dims[1])
        return flatchain, parnames[ind_show], flatlnprob

    return flatchain, parnames[ind_show]


def compute_sigma_level(trace1, trace2, nbins=30, weights=None, extents=None, **extras):
    """From a set of traces in two parameters, make a 2-d histogram of number
    of standard deviations.  Following examples from J Vanderplas.
    """
    L, xbins, ybins = np.histogram2d(trace1, trace2, bins=nbins,
                                     weights=weights,
                                     range=extents)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def figgrid(ny, nx, figsize=None, left=0.1, right=0.85,
            top=0.9, bottom=0.1, wspace=0.2, hspace=0.10):
    """Gridpars is
    left, right
    """
    from matplotlib import gridspec
    if figsize is None:
        figsize = (nx*4.5, ny*3)
    fig = pl.figure(figsize=figsize)
    axarray = np.zeros([ny, nx], dtype=np.dtype('O'))
    gs1 = gridspec.GridSpec(ny, nx)
    gs1.update(left=left, right=right, top=top, bottom=bottom,
               wspace=wspace, hspace=hspace)
    for i in range(ny):
        for j in range(nx):
            axarray[i, j] = fig.add_subplot(gs1[i, j])
    return fig, axarray


def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else pl.gca()
    ax.fill_between(x, y1, y2, **kwargs)
    p = pl.Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)
    return p


def logify(x):
    return np.log10(x)
