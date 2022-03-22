#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Much of this code adapted from dynesty.plotting (Josh Speagle)

import numpy as np
from scipy.ndimage import gaussian_filter as norm_kde

try:
    import matplotlib.pyplot as pl
    from matplotlib.ticker import MaxNLocator, NullLocator
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.colors import LinearSegmentedColormap, colorConverter
except(ImportError):
    pass


__all__ = ["allcorner", "show_extras", "prettify_axes", "corner",
           "twodhist", "marginal", "scatter",
           "get_spans", "quantile", "_quantile", "get_cmap"]


def allcorner(samples, labels, axes, weights=None, span=None,
              smooth=0.02, color="grey", qcolor=None, show_titles=False,
              hist_kwargs={"alpha": 0.5, "histtype": "stepfilled"},
              prettify=True, upper=False,
              hist2d_kwargs={}, max_n_ticks=3,
              label_kwargs={"fontsize": 12}, tick_kwargs={"labelsize": 8},
              psamples=None, samples_kwargs={"marker": "o", "color": "k"}):
    """
    Make a pretty corner plot from (weighted) posterior samples, with KDE smoothing.
    Adapted from dyensty.plotting

    Parameters
    ----------
    samples : ndarry of shape (ndim, nsamples)
        The samples of the posterior to plot

    labels : iterable of strings, with shape (ndim,)
        The labels for each dimension.

    axes : ndarray of shape (ndim, ndim)
        A 2-d array of matplotlib.pyplot.axes objects, into wich the marginal
        and joint posteriors will be plotted.

    weights : ndarray of shape (nsamples,), optional
        The weights associated with each sample.  If omitted, all samples are
        assumed to have the same weight.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D and 2-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.

    color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'black'`.

    qcolor : str or None
        If not None, plot quantiles on the marginal plots as dashed lines with
        this color.

    show_titles : bool, default=False, optional
       If True, show titles above each marginals giving median +/- numbers

    hist_kwargs : dict, optional
        Extra keyword arguments to send to the 1-D (smoothed) histograms.

    hist2d_kwargs : dict, optional
        Extra keyword arguments to send to the 2-D (smoothed) histograms.

    max_n_ticks : see `prettify_axes`

    label_kwargs : see `prettify_axes`

    tick_kwargs : see `prettify_axes`
    """
    axes = corner(samples, axes, weights=weights, span=span,
                  smooth=smooth, color=color, upper=upper,
                  hist_kwargs=hist_kwargs, hist2d_kwargs=hist2d_kwargs)

    if prettify:
        prettify_axes(axes, labels, max_n_ticks=max_n_ticks, upper=upper,
                      label_kwargs=label_kwargs, tick_kwargs=tick_kwargs)

    if psamples is not None:
        scatter(psamples, axes, zorder=10, upper=upper, **samples_kwargs)

    if (qcolor is not None) | show_titles:
        show_extras(samples, labels, axes, weights=weights,
                    qcolor=qcolor, show_titles=show_titles)

    return axes


def show_extras(samples, labels, paxes, weights=None,
                quantiles=[0.16, 0.5, 0.84], qcolor="k",
                truths=None, show_titles=False, title_fmt=".2f",
                truth_kwargs={}, title_kwargs={}):
    """Plot quantiles and truths as horizontal & vertical lines on an existing
    cornerplot.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.16, 0.5, 0.84]`
        (spanning the 68%/1-sigma credible interval).
    """
    for i, xx in enumerate(samples):
        x = xx.flatten()
        ax = paxes[i, i]
        # Plot quantiles.
        if (qcolor is not None) and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=qcolor)
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], **truth_kwargs)

        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = _quantile(x, [0.16, 0.5, 0.84], weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

        for j, yy in enumerate(samples[:i]):
            if j >= i:
                continue
            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        [ax.axvline(t, **truth_kwargs)
                         for t in truths[j]]
                    except:
                        ax.axvline(truths[j], **truth_kwargs)
                if truths[i] is not None:
                    try:
                        [ax.axhline(t, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axhline(truths[i], **truth_kwargs)


def prettify_axes(paxes, labels=None, label_kwargs={}, tick_kwargs={},
                  upper=False, max_n_ticks=3, top_ticks=False, use_math_text=True):
    """Set up cornerplot axis labels and ticks to look nice.

    Parameters
    ----------
    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    top_ticks : bool, optional (deprecated)
        Whether to label the top (rather than bottom) ticks. Default is
        `False`.

    upper : bool, optional (default=False)
        Set to true if the supplied axes should be formatted for an upper
        triangular corner plot.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.
    """

    if top_ticks:
        raise(NotImplementedError("Top ticks not implemented in corner.prettify_axes"))

    ndim = len(paxes)
    for i in range(ndim):
        for j in range(ndim):
            ax = paxes[i, j]

            if ((j < i) & upper) or ((j > i) & (not upper)):
                # suppress upper triangular or lower triangular aces
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Set up default tick spacing and format
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))

            ax.set_frame_on(True)
            if ((j == i) & upper) or ((i == (ndim - 1)) & (not upper)):
                # x-axis format for bottom valid plot in each column
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
                ax.tick_params(axis='both', which='major', **tick_kwargs)
                # TODO: this needs a separate if statement to choose the upper axes in each column
                #if top_ticks:
                #    ax.xaxis.set_ticks_position("top")
                #    [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])

            if ((j == (ndim-1)) & (i < (ndim-1)) & upper) or ((j == 0) & (i > 0) & (not upper)):
                # ylabels for edges
                ax.set_ylabel(labels[i], **label_kwargs)
                if upper:
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                else:
                    ax.yaxis.set_label_coords(-0.3, 0.5)
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.tick_params(axis='both', which='major', **tick_kwargs)
            else:
                ax.set_yticklabels([])

            if j == i:
                # diagonal are marginals, suppress y-axis ticks
                ax.yaxis.set_major_locator(NullLocator())

    return paxes


def corner(samples, paxes, weights=None, span=None, smooth=0.02,
           color='black', hist_kwargs={}, hist2d_kwargs={}, upper=False):
    """Make a smoothed cornerplot.

    :param samples: `~numpy.ndarray` of shape (ndim, nsample)
        The samples from which to construct histograms.

    :param paxes: ndarray of pyplot.Axes of shape(ndim, ndim)
        Axes into which to plot the histograms.

    :param weights: ndarray of shape (nsample,), optional
        Weights associated with each sample.

    :param span: iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    :param smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D and 2-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.

    :param color: str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'black'`.

    upper : bool, optional (default=False)
        Set to true if the supplied axes should be formatted for an upper
        triangular corner plot.

    :param hist_kwargs: dict, optional
        Extra keyword arguments to send to the 1-D (smoothed) histograms.

    :param hist2d_kwargs: dict, optional
        Extra keyword arguments to send to the 2-D (smoothed) histograms.

    :returns paxes:
    """
    assert samples.ndim > 1
    assert np.product(samples.shape[1:]) > samples.shape[0]
    ndim = len(samples)

    # Determine plotting bounds.
    span = get_spans(span, samples, weights=weights)

    # Setting up smoothing.
    smooth = np.zeros(ndim) + smooth

    # --- Now actually do the plotting-------
    for i, xx in enumerate(samples):
        x = xx.flatten()
        sx = smooth[i]

        # ---- Diagonal axes -----
        ax = paxes[i, i]
        marginal(x, ax, weights=weights, span=span[i], smooth=sx,
                 color=color, **hist_kwargs)

        # --- Off-diagonal axis ----
        for j, yy in enumerate(samples):
            y = yy.flatten()
            ax = paxes[i, j]
            if (j >= i) & (not upper):
                continue
            elif (j <= i) & upper:
                continue

            sy = smooth[j]
            twodhist(y, x, weights=weights, ax=ax,
                     span=[span[j], span[i]], smooth=[sy, sx],
                     color=color, **hist2d_kwargs)

    return paxes


def twodhist(x, y, ax=None, span=None, weights=None,
             smooth=0.02, levels=None, color='gray',
             plot_density=False, plot_contours=True, fill_contours=True,
             contour_kwargs={}, contourf_kwargs={}, **kwargs):
    """Function called by :meth:`cornerplot` used to generate a 2-D histogram
    or contour of samples.

    Parameters
    ----------
    x : interable with shape (nsamps,)
       Sample positions in the first dimension.

    y : iterable with shape (nsamps,)
       Sample positions in the second dimension.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    weights : iterable with shape (nsamps,)
        Weights associated with the samples. Default is `None` (no weights).

    levels : iterable, optional
        The contour levels to draw. Default are `[0.5, 1, 1.5, 2]`-sigma.

    ax : `~matplotlib.axes.Axes`, optional
        An `~matplotlib.axes.axes` instance on which to add the 2-D histogram.
        If not provided, a figure will be generated.

    color : str, optional
        The `~matplotlib`-style color used to draw lines and color cells
        and contours. Default is `'gray'`.

    plot_density : bool, optional
        Whether to draw the density colormap. Default is `False`.

    plot_contours : bool, optional
        Whether to draw the contours. Default is `True`.

    fill_contours : bool, optional
        Whether to fill the contours. Default is `True`.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.
    """
    # Determine plotting bounds.
    span = get_spans(span, [x, y], weights=weights)
    # Setting up smoothing.
    smooth = np.zeros(2) + smooth

    # --- Now actually do the plotting-------

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    contour_cmap = get_cmap(color, levels)
    # Initialize smoothing.
    smooth = np.zeros(2) + np.array(smooth)
    bins = []
    svalues = []
    for s in smooth:
        if s > 1.0:
            # If `s` > 1.0, the weighted histogram has
            # `s` bins within the provided bounds.
            bins.append(int(s))
            svalues.append(0.)
        else:
            # If `s` < 1, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results.
            bins.append(int(round(2. / s)))
            svalues.append(2.)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, span)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range.")

    # Smooth the results.
    if not np.all(svalues == 0.):
        H = norm_kde(H*1.0, svalues)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        print("Too few points to create valid contours.")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    clevels = np.concatenate([[0], V, [H.max() * (1 + 1e-4)]])
    # plot contour fills
    if plot_contours and fill_contours and (ax is not None):
        cfk = {}
        cfk["colors"] = contour_cmap
        cfk["antialiased"] = False
        cfk.update(contourf_kwargs)
        ax.contourf(X2, Y2, H2.T, clevels, **cfk)

    # Plot the contour edge colors.
    if plot_contours and (ax is not None):
        ck = {}
        ck["colors"] = color
        ck.update(contour_kwargs)
        ax.contour(X2, Y2, H2.T, V, **ck)

    return X2, Y2, H2.T, V, clevels, ax


def marginal(x, ax=None, weights=None, span=None, smooth=0.02,
             color='black', peak=None, **hist_kwargs):
    """Compute a marginalized (weighted) histogram, with smoothing.
    """

    if span is None:
        span = get_spans(span, np.atleast_2d(x), weights=weights)[0]
    ax.set_xlim(span)

    # Generate distribution.
    if smooth > 1:
        # If `sx` > 1, plot a weighted histogram
        xx, bins, wght = x, int(round(smooth)), weights
    else:
        # If `sx` < 1, oversample the data relative to the
        # smoothing filter by a factor of 10, then use a Gaussian
        # filter to smooth the results.
        bins = int(round(10. / smooth))
        n, b = np.histogram(x, bins=bins, weights=weights,
                            range=np.sort(span))
        n = norm_kde(n*1.0, 10.)
        b0 = 0.5 * (b[1:] + b[:-1])
        xx, bins, wght = b0, b, n

    n, b = np.histogram(xx, bins=bins, weights=wght, range=np.sort(span))
    if peak is not None:
        wght = wght * peak / n.max()

    n, b, _ = ax.hist(xx, bins=bins, weights=wght, range=np.sort(span),
                      color=color, **hist_kwargs)
    ax.set_ylim([0., max(n) * 1.05])


def scatter(samples, paxes, upper=False, **scatter_kwargs):
    """Overplot selected points on cornerplot.

    Parameters
    ----------
    samples : shape (ndim, npts)
    """
    assert samples.ndim > 1
    for i, xx in enumerate(samples):
        x = xx.flatten()
        for j, yy in enumerate(samples):
            if ((j >= i) and (not upper)) or ((j <= i) and upper):
                continue
            ax = paxes[i, j]
            y = yy.flatten()
            ax.scatter(y, x, **scatter_kwargs)


def get_spans(span, samples, weights=None):
    """Get ranges from percentiles of samples

    Parameters
    ----------
    samples : iterable of arrays
        A sequence of arrays, one for each parameter.

    Returns
    -------
    span : list of 2-tuples
        A list of (xmin, xmax) for each parameter
    """
    ndim = len(samples)
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != len(samples):
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except(TypeError):
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)
    return span


def quantile(xarr, q, weights=None):
    """Compute (weighted) quantiles from an input set of samples.

    :param x: `~numpy.darray` with shape (nvar, nsamples)
        The input array to compute quantiles of.

    :param q: list of quantiles, from [0., 1.]

    :param weights: shape (nsamples)

    :returns quants: ndarray of shape (nvar, nq)
        The quantiles of each varaible.
    """
    qq = [_quantile(x, q, weights=weights) for x in xarr]
    return np.array(qq)


def _quantile(x, q, weights=None):
    """Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """
    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def get_cmap(color, levels):
    nl = len(levels)
    from matplotlib.colors import colorConverter
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [list(rgba_color)]
    for i in range(nl+1):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)
    return contour_cmap


def demo(ndim=3, nsample=int(1e4)):
    from prospect.models.priors import Normal
    means = np.random.uniform(-3, 3, size=(ndim,))
    sigmas = np.random.uniform(1, 5, size=(ndim,))
    labels = ["x{}".format(i) for i in range(ndim)]
    prior = Normal(mean=means, sigma=sigmas)

    samples = np.array([prior.sample() for i in range(nsample)]).T
    print(samples.shape)
    print(means)
    print(sigmas)

    fig, axes = pl.subplots(ndim, ndim)
    axes = allcorner(samples, labels, axes, show_titles=True,
                     psamples=means[:, None])
    pl.show()
    return axes
