#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct, dict_to_struct

from .utils import *
from .corner import marginal

__all__ = ["pretty", "FigureMaker", "chain_to_struct"]

# some nice colors
colorcycle = ["royalblue", "firebrick", "indigo", "darkorange", "seagreen"]

# nice labels for things
pretty = {"logzsol": r"$\log (Z_{\star}/Z_{\odot})$",
          "logmass": r"$\log {\rm M}_{\star, {\rm formed}}$",
          "gas_logu": r"${\rm U}_{\rm neb}$",
          "gas_logz": r"$\log (Z_{\neb}/Z_{\odot})$",
          "dust2": r"$\tau_{\rm V}$",
          "av": r"${\rm A}_{\rm V, diffuse}$",
          "av_bc": r"${\rm A}_{\rm V, young}$",
          "dust_index": r"$\Gamma_{\rm dust}$",
          "igm_factor": r"${\rm f}_{\rm IGM}$",
          "duste_umin": r"$U_{\rm min, dust}$",
          "duste_qpah": r"$Q_{\rm PAH}$",
          "duste_gamma": r"$\gamma_{\rm dust}$",
          "log_fagn": r"$\log({\rm f}_{\rm AGN})$",
          "agn_tau": r"$\tau_{\rm AGN}$",
          "mwa": r"$\langle t \rangle_M$ (Gyr)",
          "ssfr": r"$\log ({\rm sSFR})$ $({\rm M}_\odot/{\rm yr}$",
          "tau": r"$\tau$ (Gyr)",
          "logtau": r"$\log(\tau)$ (Gyr)",
          "tage": r"Age (Gyr)",
          "ageprime": r"Age/$\tau$"}


class FigureMaker:

    show = ["mass", "logzsol", "dust2"]

    def __init__(self, results_file="", n_seds=0, prior_samples=10000,
                 nufnu=False, **extras):

        self.results_file = results_file
        self.prior_samples = prior_samples
        self.n_seds = n_seds
        self.nufnu = nufnu
        if results_file:
            self.read_in(results_file)

    def read_in(self, results_file):
        self.result, self.obs, self.model = reader.results_from(results_file)
        if self.model is None:
            raise(ValueError, "Could not build model")
        self.sps = None
        self.chain = chain_to_struct(self.result["chain"], self.model)
        self.weights = self.result.get("weights", None)
        self.ind_best = np.argmax(self.result["lnprobability"])
        self.parchain = self.convert(self.chain)

    def make_seds(self):
        if self.sps is None:
            self.build_sps()
        pbest = self.result["chain"][self.ind_best, :]
        self.spec_best, self.phot_best, _ = self.model.predict(pbest, obs=self.obs, sps=self.sps)
        if self.n_seds > 0:
            # --- get SED samples ---
            self.spec_samples, self.phot_samples, _ = self.draw_seds(self.n_seds)
        else:
            self.spec_samples = np.atleast_2d(self.spec_best)
            self.phot_samples = np.atleast_2d(self.phot_best)

    def draw_seds(self, n_seds):
        if self.sps is None:
            self.sps = self.build_sps()
        raw_samples = sample_posterior(self.result["chain"], self.weights, nsample=n_seds)
        sed_samples = [self.model.predict(p, obs=self.obs, sps=self.sps) for p in raw_samples]
        phots = np.array([sed[1] for sed in sed_samples])
        specs = np.array([sed[0] for sed in sed_samples])
        mfracs = np.array([sed[2] for sed in sed_samples])
        return specs, phots, mfracs

    def show_priors(self, diagonals, spans, smooth=0.05,
                    color="g", peak=0.96, **linekwargs):
        """Show priors, either using simple calls or using transformed samples.
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

    def plot_all(self):
        self.make_axes()
        self.styles()
        self.plot_sed(self.sax, self.rax)
        self.plot_posteriors(self.paxes)
        self.make_legend()

    def convert(self, chain):
        return chain

    # --- AXES, STYLES, LEGENDS ---

    def make_axes(self):
        raise(NotImplementedError)

    def styles(self, colorcycle=colorcycle):
        self.label_kwargs = {"fontsize": 16}
        self.tick_kwargs = {"labelsize": 14}
        # posteriors
        self.pkwargs = dict(color=colorcycle[0], alpha=0.65)
        self.skwargs = dict(color=colorcycle[1], alpha=0.65)
        self.ckwargs = dict(color=colorcycle[2], alpha=0.65)
        # histogram posteriors
        self.hkwargs = dict(histtype="stepfilled", alpha=self.pkwargs["alpha"])
        # data points
        self.dkwargs = dict(color="k", linestyle="", linewidth=1.5, mew=1.5, marker="o", mec="k", mfc="w")
        # data lines
        self.lkwargs = dict(color="k", linestyle="-", linewidth=0.75, marker="")
        # priors
        self.rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)
        # truths
        self.tkwargs = dict(color="k", linestyle="--", linewidth=1.5, mfc="k", mec="k")

        self.make_art()

    def make_art(self):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        art = {}
        art["phot_data"] = Line2D([], [], **self.dkwargs)
        art["truth"] = Line2D([], [], **self.tkwargs)
        art["phot_post"] = Patch(**self.pkwargs)
        art["spec_post"] = Patch(**self.skwargs)
        art["all_post"] = Patch(**self.ckwargs)
        art["prior"] = Line2D([], [], **self.rkwargs)
        art["spec_data"] = Line2D([], [], **self.lkwargs)
        self.art = art

    def build_sps(self):
        print("building sps from paramfile")
        self.sps = reader.get_sps(self.result)

