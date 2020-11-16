#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from . import utils
from . import sfh
from . import corner
from . import sed

from ..io.write_results import chain_to_struct, dict_to_struct
from .figuremaker import FigureMaker, colorcycle


__all__ = ["FigureMaker", "utils", "sfh", "corner", "sed",
           "pretty", "chain_to_struct"]


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
          "mwa": r"$\langle t_{\star} \rangle_M$ (Gyr)",
          "ssfr": r"$\log ({\rm sSFR})$ $({\rm yr}^{-1})$",
          "logsfr": r'$\log({\rm SFR})$ $({\rm M}_{\odot}/{\rm yr}$)',
          "tau": r"$\tau$ (Gyr)",
          "logtau": r"$\log(\tau)$ (Gyr)",
          "tage": r"Age (Gyr)",
          "ageprime": r"Age/$\tau$",
          "sigma_smooth": r"$\sigma_v$ (km/s)"}
