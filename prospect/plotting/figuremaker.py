# -*- coding: utf-8 -*-

"""figuremaker.py - module containing a class with basic plotting functionality
and convenience methods for prospector results files.
"""

import numpy as np
try:
    import matplotlib.pyplot as pl
except(ImportError):
    pass

from ..io import read_results as reader
from ..io.write_results import chain_to_struct, dict_to_struct

from .utils import *
from .corner import marginal, allcorner, get_spans
from .sed import to_nufnu

__all__ = ["FigureMaker", "colorcycle"]

# some nice colors
colorcycle = ["royalblue", "firebrick", "indigo", "darkorange", "seagreen"]


class FigureMaker(object):

    """A class for making figures from prospector results files.
    Usually you'll want to subclass this and add specific plot making methods,
    But this class contains useful methods for generating and caching
    posterior SED predictions etc.

    :param show: list of strings
        The names of the parameters (or transformed parameters) to show in
        posterior or corner plots

    :param results_file: string

    :param n_seds: int
       Number of SED samples to generate.  if < 1, generate no SEDs.
       If 0, only generate the SEDs for the most probably sample

    :param prior_samples: int
        Number of prior samples to tale when computing the prior probability
        distributions numerically

    :param nufnu: bool
        Whether to plot fluxes in nufnu

    :param microns: bool
        Whether to plot wavelength as mictons
    """

    show = ["mass", "logzsol", "dust2"]

    def __init__(self, results_file="", show=None, nufnu=False, microns=True,
                 n_seds=-1, prior_samples=10000, **extras):

        self.results_file = results_file
        self.prior_samples = prior_samples
        self.n_seds = n_seds
        self.nufnu = nufnu
        self.microns = microns
        if results_file:
            self.read_in(results_file)
        self.spec_best = self.phot_best = None
        if show is not None:
            self.show = show

    @property
    def wave_convert(self):
        return 10**(4 * self.microns)

    def read_in(self, results_file):
        """Read a prospector results file, cache important components,
        and do any parameter transformations.  The cached attributes are:

        * `obs` - The `obs` dictionary ised for the fit
        * `model` - The model used for the fit, if it could be reconstructed.
        * `chain` - Structured array of parameter vector samples
        * `weights` - Corresponding weights for each sample
        * `ind_best` - Index of the sample with the highest posterior probability
        * `parchain` - The chain transformed to the desired derived parameters via
                       the `convert` method.

        :param results_file: string
            full path of the file with the prospector results.
        """
        self.result, self.obs, self.model = reader.results_from(results_file)
        if self.model is None:
            self.model = reader.get_model(self.results)
        self.sps = None
        self.chain = chain_to_struct(self.result["chain"], self.model)
        self.weights = self.result.get("weights", None)
        self.ind_best = np.argmax(self.result["lnprobability"])
        self.parchain = self.convert(self.chain)

    def convert(self, chain):
        """Can make parameter transormations on the chain with this method,
        which you will want to subclass for your particular needs.  For example,
        this method could be used to compute the mass-weighted age for every
        posterior sample and include that in the output structured array as the
        `mwa` column.

        :param chain: structured ndarray of shape (nsample,)
            The structured ndarray of parameter values, with column names (and
            datatypes) given by the model free parameters.

        :returns parchain: structured ndarray of shape (nsample,)
            The structured ndarray of transformed parameter values.
        """
        return chain

    def make_seds(self, full=False):
        """Generate and cache the best fit model spectrum and photometry.
        Optionally generate the spectrum and photometry for a number of
        posterior samples.

        Populates the attributes `*_best` and `*_samples` where `*` is:
        * spec
        * phot
        * sed
        * cal

        :param full: bool, optional
            If true, generate the intrinsic spextrum (`sed_*`) over the entire wavelength
            range.  The (restframe) wavelength vector will be given by
            `self.sps.wavelengths`
        """
        if self.sps is None:
            self.build_sps()

        # --- best sample ---
        xbest = self.result["chain"][self.ind_best, :]
        blob = self.model.predict(xbest, obs=self.obs, sps=self.sps)
        self.spec_best, self.phot_best, self.mfrac_best = blob
        self.cal_best = self.model._speccal.copy()
        if full:
            from copy import deepcopy
            dummy = deepcopy(self.obs)
            dummy["wavelength"] = None
            dummy["spectrum"] = None
            s, p, _ = self.model.predict(xbest, obs=dummy, sps=self.sps)
        else:
            dummy = None
        self.sed_best = self.model._sed.copy()

        # --- get SED samples ---
        if self.n_seds > 0:
            blob = self.draw_seds(self.n_seds, dummy=dummy)
            self.spec_samples, self.phot_samples, self.sed_samples, self.cal_samples = blob
        else:
            self.spec_samples = np.atleast_2d(self.spec_best)
            self.phot_samples = np.atleast_2d(self.phot_best)
            self.sed_samples = np.atleast_2d(self.sed_best)
            self.cal_samples = np.atleast_2d(self.cal_best)

    def draw_seds(self, n_seds, dummy=None):
        """Draw a number of samples from the posterior chain, and generate
        spectra and photometry for them.

        :param n_seds: int
            Number of samples to draw and generate SEDs for

        :param dummy: dict, optional
            If given, use this dictionary as the obs dictionary when generating the intrinsic SED vector.
            Useful for generating an SED over a large wavelength range
        """
        if self.sps is None:
            self.sps = self.build_sps()
        raw_samples = sample_posterior(self.result["chain"], self.weights, nsample=n_seds)
        spec, phot, cal, sed, mfrac = [], [], [], [], []
        for x in raw_samples:
            s, p, m = self.model.predict(x, obs=self.obs, sps=self.sps)
            spec.append(s)
            phot.append(p)
            mfrac.append(m)
            cal.append(self.model._speccal.copy())
            if dummy is not None:
                s, p, _ = self.model.predict(x, obs=dummy, sps=self.sps)
            sed.append(self.model._sed)

        # should make this a named tuple
        return np.array(spec), np.array(phot), np.array(sed), np.array(cal)

    def build_sps(self):
        """Build the SPS object and assign it to the `sps` attribute.  This can
        be overridden by subclasses if necessary.
        """
        print("building sps from paramfile")
        self.sps = reader.get_sps(self.result)

    def show_priors(self, diagonals, spans, smooth=0.05,
                    color="g", peak=0.96, **linekwargs):
        """Show priors, either using simple calls or using transformed samples.
        These will be overplotted on the supplied axes.

        :param diagonals: ndarray of shape (nshow,)
            The axes on which to plot prior distributions, same order as `show`
        """
        samples, _ = sample_prior(self.model, nsample=self.prior_samples)
        priors = chain_to_struct(samples, self.model)
        params = self.convert(priors)
        smooth = np.zeros(len(diagonals)) + np.array(smooth)
        peak = np.zeros(len(diagonals)) + np.array(peak)
        for i, p in enumerate(self.show):
            ax = diagonals[i]
            if p in priors.dtype.names:
                x, y = get_simple_prior(self.model.config_dict[p]["prior"], spans[i])
                ax.plot(x, y * ax.get_ylim()[1] * peak[i], color=color, **linekwargs)
            else:
                marginal(params[p], ax, span=spans[i], smooth=smooth[i],
                         color=color, histtype="step", peak=ax.get_ylim()[1]*peak[i], **linekwargs)

    # --- EXAMPLE PLOTS ---

    def plot_all(self):
        """Main plotting function; makes axes, plotting styles, and then a
        corner plot and an SED (and residual) plot.
        """
        self.make_axes()
        self.styles()
        self.plot_corner(self.caxes)
        if self.nseds >= 0:
            self.make_seds()
        self.plot_sed(self.sax, self.rax, nufnu=self.nufnu, microns=self.microns)
        self.sax.legend(loc="lower right")
        self.show_transcurves(self.sax, logify=False, height=0.1)

    def plot_corner(self, caxes, **extras):
        """Example to make a corner plot of the posterior PDFs for the
        parameters listed in `show`.

        :param caxes: ndarray of axes of shape (nshow, nshow)
        """
        xx = np.squeeze(np.array(self.parchain[p] for p in self.show))
        labels = [pretty.get(p, p) for p in self.show()]
        spans = get_spans(None, xx, weights=self.weights)
        caxes = allcorner(xx, labels, caxes, weights=self.weights, span=spans,
                          color=self.pkwargs["color"], hist_kwargs=self.hkwargs,
                          label_kwargs=self.label_kwargs,
                          tick_kwargs=self.tick_kwargs, max_n_ticks=4, **extras)
        # plot priors
        if self.prior_samples > 0:
            self.show_priors(np.diag(caxes), spans, smooth=0.05, **self.rkwargs)

    def plot_sed(self, sedax, residax=None, normalize=False,
                 nufnu=True, microns=False):
        """A very basic plot of the observed photometry and the best fit
        photometry and spectrum.
        """
        # --- Data ---
        pmask = self.obs["phot_mask"]
        ophot, ounc = self.obs["maggies"][pmask], self.obs["maggies_unc"][pmask]
        owave = np.array([f.wave_effective for f in self.obs["filters"]])[pmask]
        phot_width = np.array([f.effective_width for f in self.obs["filters"]])[pmask]
        if nufnu:
            _, ophot = to_nufnu(owave, ophot, microns=microns)
            owave, ounc = to_nufnu(owave, ounc, microns=microns)
        if normalize:
            renorm = 1 / np.mean(ophot)
        else:
            renorm = 1.0

        # models
        pwave, phot_best = self.obs["phot_wave"][pmask], self.phot_best[pmask]
        spec_best = self.spec_best
        swave = self.obs.get("wavelength", None)
        if swave is None:
            if "zred" in self.model.free_params:
                zred = self.chain["zred"][self.ind_best]
            else:
                zred = self.model.params["zred"]
            swave = self.sps.wavelengths * (1 + zred)
        if nufnu:
            swave, spec_best = to_nufnu(swave, spec_best, microns=microns)
            pwave, phot_best = to_nufnu(pwave, phot_best, microns=microns)

        # plot SED
        sedax.plot(pwave, phot_best * renorm, marker="o", linestyle="",
                   **self.pkwargs, label=r"Best-fit photometry")
        sedax.plot(swave, spec_best * renorm, **self.lkwargs,
                   label=r"Best-fit spectrum")
        sedax.plot(owave, ophot * renorm, **self.dkwargs)

        # plot residuals
        if residax is not None:
            chi_phot = (ophot - phot_best) / ounc
            residax.plot(owave, chi_phot, **self.dkwargs)

    def show_transcurves(self, ax, height=0.2, logify=True,
                         linekwargs=dict(lw=1.5, color='0.3', alpha=0.7)):
        """Overplot transmission curves on an axis.  The hight of the curves
        is computed as a (logarithmic) fraction of the current plot limits.
        """
        ymin, ymax = ax.get_ylim()
        if logify:
            dyn = 10**(np.log10(ymin)+(np.log10(ymax)-np.log10(ymin))*height)
        else:
            dyn = height * (ymax-ymin)
        for f in self.obs['filters']:
            ax.plot(f.wavelength, f.transmission/f.transmission.max()*dyn+ymin,
                    **linekwargs)

    # --- AXES, STYLES, LEGENDS ---

    def restframe_axis(self, ax, microns=True, fontsize=16, ticksize=12):
        """Add a second (top) x-axis with rest-frame wavelength
        """
        if "zred" in self.model.free_params:
            zred = self.model.params["zred"]
        else:
            zred = self.parchain["zred"][self.ind_best]
        y1, y2 = ax.get_ylim()
        x1, x2 = ax.get_xlim()
        ax2 = ax.twiny()
        ax2.set_xlim(x1 / (1 + zred), x2 / (1 + zred))
        unit = microns*r"$\mu$m" + (not microns)*r"$\AA$"
        ax2.set_xlabel(r'$\lambda_{{\rm rest}}$ ({})'.format(unit), fontsize=fontsize)
        ax2.set_ylim(y1, y2)
        ax2.tick_params('both', pad=2.5, size=3.5, width=1.0, which='both', labelsize=ticksize)

    def make_axes(self):
        """Make a set of axes and assign them to the object.
        """
        self.caxes = pl.subplots(len(self.show), len(self.show))

    def styles(self, colorcycle=colorcycle):
        """Define a set of plotting styles for use throughout the figure.
        """
        self.label_kwargs = {"fontsize": 16}
        self.tick_kwargs = {"labelsize": 14}
        # posteriors
        self.pkwargs = dict(color=colorcycle[0], alpha=0.65)
        # histogram posteriors
        self.hkwargs = dict(histtype="stepfilled", alpha=self.pkwargs["alpha"])
        # data points
        self.dkwargs = dict(color="k", linestyle="", linewidth=1.5, markersize=6,
                            mew=2, marker="o", mec="k", mfc="gray")
        # data lines
        self.lkwargs = dict(color="k", linestyle="-", linewidth=0.75, marker="")
        # priors
        self.rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)
        # truths
        self.tkwargs = dict(color="k", linestyle="--", linewidth=1.5, mfc="k", mec="k")

        self.make_art()

    def make_art(self):
        """Make a dictionary of artists corresponding to the plotting styles
        that can be used for making legends
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        art = {}
        art["phot_data"] = Line2D([], [], **self.dkwargs)
        art["truth"] = Line2D([], [], **self.tkwargs)
        art["posterior"] = Patch(**self.pkwargs)
        art["prior"] = Line2D([], [], **self.rkwargs)
        art["spec_data"] = Line2D([], [], **self.lkwargs)
        self.art = art

    def make_legend(self):
        raise(NotImplementedError)


if __name__ == "__main__":
    pass