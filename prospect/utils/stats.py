#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
try:
    import matplotlib.pyplot as pl
except(ImportError):
    pass


__all__ = ["get_best", "get_truths", "get_percentiles", "get_stats",
           "posterior_samples", "hist_samples", "joint_pdf", "compute_sigma_level",
           "trim_walkers", "fill_between", "figgrid"]


def flatstruct(struct):
    params = struct.dtype.names
    m = [struct[s] for s in params]
    return np.concatenate(m), params


def get_best(res, **kwargs):
    """Get the maximum a posteriori parameters and their names

    :param res:
        A ``results`` dictionary with the keys 'lnprobability', 'chain', and
        'theta_labels'

    :returns theta_names:
        List of strings giving the names of the parameters, of length ``ndim``

    :returns best:
        ndarray with shape ``(ndim,)`` of parameter values corresponding to the
        sample with the highest posterior probaility
    """
    qbest, qnames = flatstruct(best_sample(res))
    return qnames, qbest


def best_sample(res):
    """Get the posterior sample with the highest posterior probability.
    """
    imax = np.argmax(res['lnprobability'])
    # there must be a more elegant way to deal with differnt shapes
    try:
        i, j = np.unravel_index(imax, res['lnprobability'].shape)
        Qbest = res['chain'][i, j].copy()
    except(ValueError):
        Qbest = res['chain'][imax].copy()
    return Qbest


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

    chaincat = res["chain"]
    niter = res['chain'].shape[0]
    parnames = chaincat.dtype.names
    weights = res.get("weights", None)


    start_index = np.floor(start * (niter-1)).astype(int)
    if res["chain"].ndim > 1:
        flatchain = res['chain'][:, start_index::thin]
        flatchain = flatchain.reshape(-1)
    elif res["chain"].ndim == 1:
        flatchain = res["chain"][start_index::thin]
    chain = flatstruct(chaincat)

    pct = [quantile(x, ptile, weights=weights, axis=0)
           for x in chain.T]
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


def posterior_samples(res, nsample=None, **kwargs):
    """Pull samples of theta from the MCMC chain

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param nsample:
        Number of random samples to draw.

    :param **kwargs:
        Extra keywords are passed to ``hist_samples``.

    :returns thetas:
        A list of parameter vectors pulled at random from the chain, of same
        length as ``samples``.
    """
    flatchain, pnames = hist_samples(res, **kwargs)
    weights = res.get("weights", None)
    ns = flatchain.shape[0]
    if nsample is None:
        nsample = ns
    s = np.random.choice(ns, p=weights, size=nsample)
    thetas = flatchain[s, :]
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
        # emcee
        flatchain = res['chain'][:, start_index::thin, :]
        dims = flatchain.shape
        flatchain = flatchain.reshape(dims[0]*dims[1], dims[2])
        flatlnprob = res['lnprobability'][:, start_index::thin].reshape(dims[0]*dims[1])
    elif res["chain"].ndim == 2:
        # dynesty
        flatchain = res["chain"][start_index::thin, :]
        flatlnprob = res['lnprobability'][start_index::thin]
    if showpars is None:
        ind_show = slice(None)
    else:
        ind_show = np.array([p in showpars for p in parnames], dtype=bool)
    flatchain = flatchain[:, ind_show]
    if return_lnprob:
        return flatchain, parnames[ind_show], flatlnprob

    return flatchain, parnames[ind_show]


def logify(x):
    return np.log10(x)
