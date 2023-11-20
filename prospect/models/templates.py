#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""templates.py -- A set of predefined "base" prospector model specifications
that can be used as a starting point and then combined or altered.
"""

from copy import deepcopy
import numpy as np
import os
from . import priors
from . import priors_beta
from . import transforms

__all__ = ["TemplateLibrary",
           "describe",
           "adjust_dirichlet_agebins",
           "adjust_continuity_agebins",
           ]


class Directory(object):
    """A dict-like that only returns copies of the dictionary values.
    It also includes a dictionary of information describing each entry in the
    directory.
    """

    def __init__(self):
        self._entries = {}
        self._descriptions = {}
        try:
            self.iteritems = self._entries.iteritems
        except AttributeError:
            self.iteritems = self._entries.items

    def __getitem__(self, k):
        return deepcopy(self._entries[k])

    def __setitem__(self, k, v):
        entry, description = v
        self._entries[k] = entry
        self._descriptions[k] = description

    def describe(self, k):
        print(describe(self._entries[k]))

    def show_contents(self):
        for k, v in list(self._descriptions.items()):
            print("'{}':\n  {}".format(k, v))


def describe(parset, current_params={}):
    ttext = "Free Parameters: (name: prior) \n-----------\n"
    free = ["{}: {}".format(k, v["prior"])
            for k, v in list(parset.items()) if v.get("isfree", False)]
    ttext += "  " + "\n  ".join(free)

    ftext = "Fixed Parameters: (name: value [, depends_on]) \n-----------\n"
    fixed = ["{}: {} {}".format(k, current_params.get(k, v["init"]),
                                v.get("depends_on", ""))
             for k, v in list(parset.items()) if not v.get("isfree", False)]
    ftext += "  " + "\n  ".join(fixed)
    return ttext + "\n\n" + ftext


def adjust_dirichlet_agebins(parset, agelims=[0., 8., 9., 10.]):
    """Given a list of limits in age for bins, adjust the parameter
    specifications to work for those limits.

    :param parset:
        The parameter specification dictionary to adjust.  Must have entries (keys) for
        "mass", "agebins", "zfraction"

    :param agelims:
        An iterable fo bin edges, in log(yrs) of lookback time.
    """
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    ncomp = len(agelims) - 1
    # constant SFR
    zinit = np.array([(i-1)/float(i) for i in range(ncomp, 1, -1)])

    # Set up the prior in `z` variables that corresponds to a dirichlet in sfr
    # fraction.  THIS IS IMPORTANT
    alpha = np.arange(ncomp-1, 0, -1)
    zprior = priors.Beta(alpha=alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    parset['mass']['N'] = ncomp
    parset['agebins']['N'] = ncomp
    parset['agebins']['init'] = agebins
    parset['z_fraction']['N'] = len(zinit)
    parset['z_fraction']['init'] = zinit
    parset['z_fraction']['prior'] = zprior

    return parset


def adjust_continuity_agebins(parset, tuniv=13.7, nbins=7):
    """defines agebins
    the first two agebins are hard-coded to be 0-30 Myr, 30-100 Myr
    the final agebin is hard-coded to cover 0.85*t_univ-t_univ
    the rest split logarithmic time evenly

    inputs:
        tuniv is the age of the Universe in Gyr
        nbins is the number of SFH bins
    """

    if nbins < 4:
        raise ValueError('Must have nbins >= 4, returning')

    tbinmax = (tuniv * 0.85) * 1e9
    lim1, lim2 = 7.4772, 8.0
    agelims = ([0, lim1] +
               np.linspace(lim2, np.log10(tbinmax), nbins-2).tolist() +
               [np.log10(tuniv*1e9)])
    agebins = np.array([agelims[:-1], agelims[1:]])

    ncomp = nbins
    mean = np.zeros(ncomp-1)
    scale = np.ones_like(mean) * 0.3
    df = np.ones_like(mean) * 2
    rprior = priors.StudentT(mean=mean, scale=scale, df=df)

    parset['mass']['N'] = ncomp
    parset['agebins']['N'] = ncomp
    parset['agebins']['init'] = agebins.T
    parset["logsfr_ratios"]["N"] = ncomp - 1
    parset["logsfr_ratios"]["init"] = mean
    parset["logsfr_ratios"]["prior"] = rprior

    return parset


TemplateLibrary = Directory()

# A template for what parameter configuration element should look like
par_name = {"N": 1,
            "isfree": True,
            "init": 0.5,
            "prior": priors.TopHat(mini=0, maxi=1.0),
            "depends_on": None,
            "units": ""}

# ---------------------
# --- Explicit defaults
# --------------------

imf = {"N": 1, "isfree": False, "init": 2}        # Kroupa
dust_type = {"N": 1, "isfree": False, "init": 0}  # Power-law

_defaults_ = {"imf_type": imf,        # FSPS parameter
              "dust_type": dust_type  # FSPS parameter
              }

TemplateLibrary["type_defaults"] = (_defaults_,
                                    "Explicitly sets dust amd IMF types.")

# --------------------------
# --- Some (very) common parameters ----
# --------------------------

zred = {"N": 1, "isfree": False,
        "init": 0.1,
        "units": "redshift",
        "prior": priors.TopHat(mini=0.0, maxi=4.0)}

mass = {"N": 1, "isfree": True,
        "init": 1e10,
        "units": "Solar masses formed",
        "prior": priors.LogUniform(mini=1e8, maxi=1e12)}

logzsol = {"N": 1, "isfree": True,
           "init": -0.5,
           "units": r"$\log (Z/Z_\odot)$",
           "prior": priors.TopHat(mini=-2, maxi=0.19)}

dust2 = {"N": 1, "isfree": True,
         "init": 0.6,
         "units": "optical depth at 5500AA",
         "prior": priors.TopHat(mini=0.0, maxi=2.0)}

sfh = {"N": 1, "isfree": False, "init": 0, "units": "FSPS index"}

tage = {"N": 1, "isfree": True,
        "init": 1, "units": "Gyr",
        "prior": priors.TopHat(mini=0.001, maxi=13.8)}

_basic_ = {"zred": zred,
           "mass": mass,
           "logzsol": logzsol,  # FSPS parameter
           "dust2": dust2,      # FSPS parameter
           "sfh": sfh,          # FSPS parameter
           "tage": tage         # FSPS parameter
           }

_basic_.update(_defaults_)

TemplateLibrary["ssp"] = (_basic_,
                          ("Basic set of (free) parameters for a delta function SFH"))


# ----------------------------
# --- Parametric SFH -----
# ----------------------------
_parametric_ = TemplateLibrary["ssp"]
_parametric_["sfh"]["init"] = 4   # Delay-tau
_parametric_["tau"] = {"N": 1, "isfree": True,
                       "init": 1, "units": "Gyr^{-1}",
                       "prior": priors.LogUniform(mini=0.1, maxi=30)}

TemplateLibrary["parametric_sfh"] = (_parametric_,
                                     ("Basic set of (free) parameters for a delay-tau SFH."))


# --------------------------
# ---  Dust emission ----
# --------------------------
add_duste = {"N": 1, "isfree": False, "init": True}

duste_umin = {"N": 1, "isfree": False,
              "init": 1.0, "units": 'MMP83 local MW intensity',
              "prior": priors.TopHat(mini=0.1, maxi=25)}

duste_qpah = {"N": 1, "isfree": False,
              "init": 4.0, "units": 'Percent mass fraction of PAHs in dust.',
              "prior": priors.TopHat(mini=0.5, maxi=7.0)}


duste_gamma = {"N": 1, "isfree": False,
               "init": 1e-3, "units": 'Mass fraction of dust in high radiation intensity.',
               "prior": priors.LogUniform(mini=1e-3, maxi=0.15)}

_dust_emission_ = {"add_dust_emission": add_duste,
                   "duste_umin": duste_umin,    # FSPS / Draine & Li parameter
                   "duste_qpah": duste_qpah,    # FSPS / Draine & Li parameter
                   "duste_gamma": duste_gamma   # FSPS / Draine & Li parameter
                   }

TemplateLibrary["dust_emission"] = (_dust_emission_,
                                    ("The set of (fixed) dust emission parameters."))

# --------------------------
# --- Nebular Emission ----
# --------------------------

add_neb = {'N': 1, 'isfree': False, 'init': True}
neb_cont = {'N': 1, 'isfree': False, 'init': True}
neb_spec = {'N': 1, 'isfree': False, 'init': True}

# Note this depends on stellar metallicity
gas_logz = {'N': 1, 'isfree': False,
            'init': 0.0, 'units': r'log Z/Z_\odot',
            'depends_on': transforms.stellar_logzsol,
            'prior': priors.TopHat(mini=-2.0, maxi=0.5)}

gas_logu = {"N": 1, 'isfree': False,
            'init': -2.0, 'units': r"Q_H/N_H",
            'prior': priors.TopHat(mini=-4, maxi=-1)}

_nebular_ = {"add_neb_emission": add_neb,    # FSPS parameter.
             "add_neb_continuum": neb_cont,  # FSPS parameter.
             "nebemlineinspec": neb_spec,    # FSPS parameter.
             "gas_logz": gas_logz,           # FSPS parameter.
             "gas_logu": gas_logu,           # FSPS parameter.
             }

TemplateLibrary["nebular"] = (_nebular_,
                              ("The set of nebular emission parameters, "
                               "with gas_logz tied to stellar logzsol."))

# -----------------------------------------
# --- Nebular Emission Marginalization ----
# -----------------------------------------
marginalize_elines = {'N': 1, 'isfree': False, 'init': True}
use_eline_prior = {'N': 1, 'isfree': False, 'init': True}
nebemlineinspec = {'N': 1, 'isfree': False, 'init': False}  # can't be included w/ marginalization

# marginalize over which of the 128 FSPS emission lines?
# input is a list of emission line names matching $SPS_HOME/data/emlines_info.dat
SPS_HOME = os.getenv('SPS_HOME')
try:
    info = np.genfromtxt(os.path.join(SPS_HOME, 'data', 'emlines_info.dat'),
                     dtype=[('wave', 'f8'), ('name', '<U20')],
                     delimiter=',')
except OSError:
    info = {'name':[]}
except TypeError:
    # SPS_HOME not defined
    info = {'name':[]}

# Fit all lines by default
elines_to_fit = {'N': 1, 'isfree': False, 'init': np.array(info['name'])}
eline_prior_width = {'N': 1, 'isfree': False,
                     'init': 0.2,
                     'units': r'width of Gaussian prior on line luminosity, in units of (true luminosity/FSPS predictions)',
                     'prior': None}

eline_delta_zred = {'N': 1, 'isfree': True,
                    'init': 0.0, 'units': r'redshift',
                    'prior': priors.TopHat(mini=-0.01, maxi=0.01)}

eline_sigma = {'N': 1, 'isfree': True,
               'init': 100.0, 'units': r'km/s',
               'prior': priors.TopHat(mini=30, maxi=300)}

_neb_marg_ = {"marginalize_elines": marginalize_elines,
              "use_eline_prior": use_eline_prior,
              "nebemlineinspec": nebemlineinspec,
              "elines_to_fit": elines_to_fit,
              "eline_prior_width": eline_prior_width,
              "eline_sigma": eline_sigma
              }

_fit_eline_redshift_ = {'eline_delta_zred': eline_delta_zred}

TemplateLibrary["nebular_marginalization"] = (_neb_marg_,
                                              ("Marginalize over emission amplitudes line contained in"
                                               "the observed spectrum"))

TemplateLibrary["fit_eline_redshift"] = (_fit_eline_redshift_,
                                              ("Fit for the redshift of the emission lines separately"
                                               "from the stellar redshift"))

# ------------------------
# --- AGN Nebular emission
# ------------------------
_agn_eline_ = {}
_agn_eline_["agn_elum"] = dict(N=1, isfree=False, init=1e-4,
                               prior=priors.Uniform(mini=1e-6, maxi=1e-2),
                               units="L_Hbeta(Lsun) / Mformed")
_agn_eline_["agn_eline_sigma"] = dict(N=1, isfree=False, init=100.0,
                                      prior=priors.Uniform(mini=50, maxi=500))
_agn_eline_["nebemlineinspec"] = dict(N=1, isfree=False, init=False)  # can't be included w/ AGN lines

TemplateLibrary["agn_eline"] = (_agn_eline_,
                                ("Add AGN emission lines"))

# -------------------------
# --- Outlier Templates ---
# -------------------------

f_outlier_spec = {"N": 1,
                  "isfree": True,
                  "init": 0.01,
                  "prior": priors.TopHat(mini=1e-5, maxi=0.5)}

nsigma_outlier_spec = {"N": 1,
                       "isfree": False,
                       "init": 50.0}

f_outlier_phot = {"N": 1,
                  "isfree": False,
                  "init": 0.00,
                  "prior": priors.TopHat(mini=0.0, maxi=0.5)}

nsigma_outlier_phot = {"N": 1,
                       "isfree": False,
                       "init": 50.0}

_outlier_modeling_ = {"f_outlier_spec": f_outlier_spec,
                      "nsigma_outlier_spec": nsigma_outlier_spec,
                      "f_outlier_phot": f_outlier_phot,
                      "nsigma_outlier_phot": nsigma_outlier_phot
                      }

TemplateLibrary['outlier_model'] = (_outlier_modeling_,
                                   ("The set of outlier (mixture) models for spectroscopy and photometry"))

# --------------------------
# --- AGN Torus Emission ---
# --------------------------
add_agn = {"N": 1, "isfree": False, "init": True}

fagn = {'N': 1, 'isfree': False,
        'init': 1e-4, 'units': r'L_{AGN}/L_*',
        'prior': priors.LogUniform(mini=1e-5, maxi=3.0)}

agn_tau = {"N": 1, 'isfree': False,
           "init": 5.0, 'units': r"optical depth",
           'prior': priors.LogUniform(mini=5.0, maxi=150.)}

_agn_ = {"fagn": fagn,       # FSPS parameter.
         "agn_tau": agn_tau,  # FSPS parameter.
         "add_agn_dust": add_agn
         }

TemplateLibrary["agn"] = (_agn_,
                          ("The set of (fixed) AGN dusty torus emission parameters."))

# --------------------------
# --- IGM Absorption ---
# --------------------------
add_igm = {'N': 1, 'isfree': False, 'init': True}

igm_fact = {'N': 1, 'isfree': False, 'init': 1.0,
            'units': 'factor by which to scale the Madau attenuation',
            'prior': priors.ClippedNormal(mean=1.0, sigma=0.1, mini=0.0, maxi=2.0)}

_igm_ = {"add_igm_absorption": add_igm,  # FSPS Parameter.
         "igm_factor": igm_fact,   # FSPS Parameter.
         }

TemplateLibrary["igm"] = (_igm_,
                          ("The set of (fixed) IGM absorption parameters."))


# --------------------------
# --- Spectral Smoothing ---
# --------------------------
smooth = {'N': 1, 'isfree': False, 'init': 'vel'}
fft =    {'N': 1, 'isfree': False, 'init': True}
wlo =    {'N': 1, 'isfree': False, 'init': 3500.0, 'units': r'$\AA$'}
whi =    {'N': 1, 'isfree': False, 'init': 7800.0, 'units': r'$\AA$'}

sigma_smooth = {'N': 1, 'isfree': True,
                'init': 200.0, 'units': 'km/s',
                'prior': priors.TopHat(mini=10, maxi=300)}

_smoothing_ = {"smoothtype": smooth, "fftsmooth": fft,   # prospecter `smoothspec` parameter
               #"min_wave_smooth": wlo, "max_wave_smooth": whi,
               "sigma_smooth": sigma_smooth  # prospecter `smoothspec` parameter
               }

TemplateLibrary["spectral_smoothing"] = (_smoothing_,
                                        ("Set of parameters for spectal smoothing."))


# --------------------------
# --- Spectral calibration
# -------------------------

spec_norm = {'N': 1, 'isfree': False,
             'init': 1.0, 'units': 'f_true/f_obs',
             'prior': priors.Normal(mean=1.0, sigma=0.1)}
# What order polynomial?
npoly = 12
porder = {'N': 1, 'isfree': False, 'init': npoly}
preg = {'N': 1, 'isfree': False, 'init': 0.}
polymax = 0.1 / (np.arange(npoly) + 1)
pcoeffs = {'N': npoly, 'isfree': True,
           'init': np.zeros(npoly),
           'units': 'ln(f_tru/f_obs)_j=\sum_{i=1}^N poly_coeffs_{i-1} * lambda_j^i',
           'prior': priors.TopHat(mini=-polymax, maxi=polymax)}

_polyopt_ = {"polyorder": porder,         # order of polynomial to optimize
             "poly_regularization": preg, # Regularization of polynomial coeffs (can be a vector).
             "spec_norm": spec_norm       # Overall normalization of the spectrum.
             }
_polyfit_ = {"spec_norm": spec_norm, # Overall normalization of the spectrum.
             "poly_coeffs": pcoeffs  # Polynomial coefficients
             }

TemplateLibrary["optimize_speccal"] = (_polyopt_,
                                       ("Set of parameters (most of which are fixed) "
                                        "for optimizing a polynomial calibration vector."))
TemplateLibrary["fit_speccal"] = (_polyfit_,
                                  ("Set of parameters (most of which are free) for sampling "
                                   "the coefficients of a polynomial calibration vector."))

# ----------------------------
# --- Additional SF Bursts ---
# ---------------------------

fage_burst = {'N': 1, 'isfree': False,
              'init': 0.0, 'units': 'time at wich burst happens, as a fraction of `tage`',
              'prior': priors.TopHat(mini=0.5, maxi=1.0)}

tburst = {'N': 1, 'isfree': False,
          'init': 0.0, 'units': 'Gyr',
          'prior': None, 'depends_on': transforms.tburst_from_fage}

fburst = {'N': 1, 'isfree': False,
          'init': 0.0, 'units': 'fraction of total mass formed in the burst',
          'prior': priors.TopHat(mini=0.0, maxi=0.5)}

_burst_ = {"tburst": tburst,
           "fburst": fburst,
           "fage_burst": fage_burst}

TemplateLibrary["burst_sfh"] = (_burst_,
                                ("The set of (fixed) parameters for an SF burst "
                                 "added to a parameteric SFH, with the burst time "
                                 "controlled by `fage_burst`."))

# -----------------------------------
# --- Nonparametric-logmass SFH ----
# -----------------------------------
# Using a (perhaps dangerously) simple nonparametric model of mass in fixed time bins with a logarithmic prior.

_nonpar_lm_ = TemplateLibrary["ssp"]
_ = _nonpar_lm_.pop("tage")

_nonpar_lm_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
nbin = 3
# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  Its length needs to be modified based on the number of bins
_nonpar_lm_["mass"]       = {'N': nbin, 'isfree': True, 'units': r'M$_\odot$',
                             'init': np.zeros(nbin) + 1e6,
                             'prior': priors.LogUniform(mini=np.zeros(nbin)+1e5, maxi=np.zeros(nbin)+1e12)}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_nonpar_lm_["agebins"]    = {'N': nbin, 'isfree': False,
                             'init': [[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]],
                             'units': 'log(yr)'}
# This is the *total* stellar mass formed
_nonpar_lm_["total_mass"] = {"N": 1, "isfree": False, "init": 1e10, "units": "Solar masses formed",
                             "depends_on": transforms.total_mass}

TemplateLibrary["logm_sfh"] = (_nonpar_lm_,
                               "Non-parameteric SFH fitting for log-mass in fixed time bins")

# ----------------------------
# --- Continuity SFH ----
# ----------------------------
# A non-parametric SFH model of mass in fixed time bins with a smoothness prior

_nonpar_continuity_ = TemplateLibrary["ssp"]
_ = _nonpar_continuity_.pop("tage")

_nonpar_continuity_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
# This is the *total*  mass formed, as a variable
_nonpar_continuity_["logmass"]    = {"N": 1, "isfree": True, "init": 10, 'units': 'Msun',
                                     'prior': priors.TopHat(mini=7, maxi=12)}
# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  Its length needs to be modified based on the number of bins
_nonpar_continuity_["mass"]       = {'N': 3, 'isfree': False, 'init': 1e6, 'units': r'M$_\odot$',
                                     'depends_on': transforms.logsfr_ratios_to_masses}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_nonpar_continuity_["agebins"]    = {'N': 3, 'isfree': False,
                                     'init': [[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]],
                                     'units': 'log(yr)'}
# This controls the distribution of SFR(t) / SFR(t+dt). It has NBINS-1 components.
_nonpar_continuity_["logsfr_ratios"] = {'N': 2, 'isfree': True, 'init': [0.0, 0.0],
                                        'prior': priors.StudentT(mean=np.full(2, 0.0),
                                                                 scale=np.full(2, 0.3),
                                                                 df=np.full(2, 2))}
TemplateLibrary["continuity_sfh"] = (_nonpar_continuity_,
                                     "Non-parameteric SFH fitting for mass in fixed time bins with a smoothness prior")

# ----------------------------
# --- Flexible Continuity SFH ----
# ----------------------------
# A non-parametric SFH model of mass in flexible time bins with a smoothness prior

_nonpar_continuity_flex_ = TemplateLibrary["ssp"]
_ = _nonpar_continuity_flex_.pop("tage")

_nonpar_continuity_flex_["sfh"] = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
#_nonpar_continuity_flex_["tuniv"]      = {"N": 1, "isfree": False, "init": 13.7, "units": "Gyr"}

# This is the *total*  mass formed
_nonpar_continuity_flex_["logmass"] = {"N": 1, "isfree": True, "init": 10, 'units': 'Msun',
                                       'prior': priors.TopHat(mini=7, maxi=12)}
# These variables control the ratio of SFRs in adjacent bins
# there is one for a fixed "youngest" bin, one for the fixed "oldest" bin, and (N-1) for N flexible bins in between
_nonpar_continuity_flex_["logsfr_ratio_young"] = {'N': 1, 'isfree': True, 'init': 0.0, 'units': r'dlogSFR (dex)',
                                                  'prior': priors.StudentT(mean=0.0, scale=0.3, df=2)}
_nonpar_continuity_flex_["logsfr_ratio_old"] = {'N': 1, 'isfree': True, 'init': 0.0, 'units': r'dlogSFR (dex)',
                                                'prior': priors.StudentT(mean=0.0, scale=0.3, df=2)}
_nonpar_continuity_flex_["logsfr_ratios"] = {'N': 1, 'isfree': True, 'init': 0.0, 'units': r'dlogSFR (dex)',
                                             'prior': priors.StudentT(mean=0.0, scale=0.3, df=2)}

# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  Its length needs to be modified based on the total number of
# bins (including fixed young and old bin)
_nonpar_continuity_flex_["mass"] = {'N': 4, 'isfree': False, 'init': 1e6, 'units': r'M$_\odot$',
                                    'depends_on': transforms.logsfr_ratios_to_masses_flex}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_nonpar_continuity_flex_["agebins"]    = {'N': 4, 'isfree': False,
                                          'depends_on': transforms.logsfr_ratios_to_agebins,
                                          'init': [[0.0, 7.5], [7.5, 8.5], [8.5, 9.7], [9.7, 10.136]],
                                          'units': 'log(yr)'}

TemplateLibrary["continuity_flex_sfh"] = (_nonpar_continuity_flex_,
                                          ("Non-parameteric SFH fitting for mass in flexible time "
                                           "bins with a smoothness prior"))

# ----------------------------
# --- PSB Continuity SFH ----
# ----------------------------
# A non-parametric SFH model of mass in Nfixed fixed bins and Nflex flexible time bins with a smoothness prior. Model described in detail in Suess et al. (2021).

_nonpar_continuity_psb_ = TemplateLibrary["ssp"]
_ = _nonpar_continuity_psb_.pop("tage")

_nonpar_continuity_psb_["sfh"] = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}

# This is the *total*  mass formed
_nonpar_continuity_psb_["logmass"] = {"N": 1, "isfree": True, "init": 10, 'units': 'Msun',
                                      "prior": priors.TopHat(mini=7, maxi=12)}

# set up the total number of bins that we want in our SFH.
# there are nfixed "oldest" bins, one "youngest" bin, and nflex flexible bins in between
# the youngest bin has variable width tlast. tflex is specified by the user, and
# sets the amount of time available for the flexible+youngest bins (e.g., t_lookback=tflex
# is the time when the SFH transitions from fixed to flexible bins)
_nonpar_continuity_psb_['tflex'] = {'name': 'tflex', 'N': 1, 'isfree': False, 'init': 2, 'units':'Gyr'}
_nonpar_continuity_psb_['nflex'] = {'name': 'nflex', 'N': 1, 'isfree': False, 'init': 5}
_nonpar_continuity_psb_['nfixed'] = {'name': 'nfixed', 'N': 1, 'isfree': False, 'init': 3}
_nonpar_continuity_psb_['tlast'] = {'name': 'tlast', 'N': 1, 'isfree': True,
                                    'init': 0.2, 'prior': priors.TopHat(mini=.01, maxi=1.5)}

# These variables control the ratio of SFRs in adjacent bins
# there is one for a fixed "youngest" bin, nfixed for nfixed "oldest" bins,
# and (nflex-1) for nflex flexible bins in between
_nonpar_continuity_psb_["logsfr_ratio_young"] = {'N': 1, 'isfree': True, 'init': 0.0, 'units': r'dlogSFR (dex)',
                                                 'prior': priors.StudentT(mean=0.0, scale=0.3, df=2)}
_nonpar_continuity_psb_["logsfr_ratio_old"] = {'N': 3, 'isfree': True, 'init': np.zeros(3), 'units': r'dlogSFR (dex)',
                                               'prior': priors.StudentT(mean=np.zeros(3), scale=np.ones(3)*0.3, df=np.ones(3))}
_nonpar_continuity_psb_["logsfr_ratios"] = {'N': 4, 'isfree': True, 'init': np.zeros(4), 'units': r'dlogSFR (dex)',
                                            'prior': priors.StudentT(mean=np.zeros(4), scale=0.3*np.ones(4), df=np.ones(4))}

# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  Its length needs to be modified based on the total number of
# bins (including fixed young and old bin)
_nonpar_continuity_psb_["mass"] = {'N': 9, 'isfree': False, 'init': 1e6, 'units': r'M$_\odot$',
                                   'depends_on': transforms.logsfr_ratios_to_masses_psb}
# This gives the start and stop of each age bin.  The fixed bins can/should be adjusted and its
# length must match the lenth of "mass"
agelims = np.array([1, 0.2*1e9] +
                   np.linspace((0.3 + .1)*1e9, 2e9, 5).tolist() +
                   np.linspace(2e9, 13.6e9, 4)[1:].tolist())
_nonpar_continuity_psb_["agebins"]    = {'N': 9, 'isfree': False,
                                         'depends_on': transforms.psb_logsfr_ratios_to_agebins,
                                         'init': np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T,
                                         'units': 'log(yr)'}

TemplateLibrary["continuity_psb_sfh"] = (_nonpar_continuity_psb_,
                                         ("Non-parameteric SFH fitting for mass in Nfixed fixed bins "
                                          "and Nflex flexible time bins with a smoothness prior"))

# ----------------------------
# --- Dirichlet SFH ----
# ----------------------------
# Using the dirichlet prior on SFR fractions in bins of constant SF.

_dirichlet_ = TemplateLibrary["ssp"]
_ = _dirichlet_.pop("tage")

_dirichlet_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  It's length needs to be modified based on the number of bins
_dirichlet_["mass"]       = {'N': 3, 'isfree': False, 'init': 1., 'units': r'M$_\odot$',
                          'depends_on': transforms.zfrac_to_masses}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_dirichlet_["agebins"]    = {'N': 3, 'isfree': False,
                             'init': [[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]],
                             'units': 'log(yr)'}
# Auxiliary variable used for sampling sfr_fractions from dirichlet. This
# *must* be adjusted depending on the number of bins
_dirichlet_["z_fraction"] = {"N": 2, 'isfree': True, 'init': [0, 0], 'units': None,
                             "prior": priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)}
# This is the *total* stellar mass formed
_dirichlet_["total_mass"] = mass

TemplateLibrary["dirichlet_sfh"] = (_dirichlet_,
                                    "Non-parameteric SFH with Dirichlet prior (fractional SFR)")

# ----------------------------
# --- Prospector-alpha ---
# ----------------------------

_alpha_ = TemplateLibrary["dirichlet_sfh"]
_alpha_.update(TemplateLibrary["dust_emission"])
_alpha_.update(TemplateLibrary["nebular"])
_alpha_.update(TemplateLibrary["agn"])

# Set the dust and agn emission free
_alpha_["duste_qpah"]["isfree"] = True
_alpha_["duste_umin"]["isfree"] = True
_alpha_["duste_gamma"]["isfree"] = True
_alpha_["fagn"]["isfree"] = True
_alpha_["agn_tau"]["isfree"] = True

# Complexify the dust attenuation
_alpha_["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
_alpha_["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
_alpha_["dust1"]      = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                         "init": 0.0, "units": "optical depth towards young stars"}

_alpha_["dust_ratio"] = {"N": 1, "isfree": True,
                         "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                         "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

_alpha_["dust_index"] = {"N": 1, "isfree": True,
                         "init": 0.0, "units": "power-law multiplication of Calzetti",
                         "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
# in Gyr
alpha_agelims = np.array([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
_alpha_ = adjust_dirichlet_agebins(_alpha_, agelims=(np.log10(alpha_agelims) + 9))

TemplateLibrary["alpha"] = (_alpha_,
                            "The prospector-alpha model, Leja et al. 2017")


# ----------------------------
# --- Prospector-beta ---
# ----------------------------

_beta_nzsfh_ = TemplateLibrary["alpha"]
_beta_nzsfh_.pop('z_fraction', None)
_beta_nzsfh_.pop('total_mass', None)

nbins_sfh = 7 # number of sfh bins
_beta_nzsfh_['nzsfh'] = {'N': nbins_sfh+2, 'isfree': True, 'init': np.concatenate([[0.5,8,0.0], np.zeros(nbins_sfh-1)]),
                         'prior': priors_beta.NzSFH(zred_mini=1e-3, zred_maxi=15.0,
                                                    mass_mini=7.0, mass_maxi=12.5,
                                                    z_mini=-1.98, z_maxi=0.19,
                                                    logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                                    logsfr_ratio_tscale=0.3, nbins_sfh=nbins_sfh,
                                                    const_phi=True)}

_beta_nzsfh_['zred'] = {'N': 1, 'isfree': False, 'init': 0.5,
                        'depends_on': transforms.nzsfh_to_zred}

_beta_nzsfh_['logmass'] = {'N': 1, 'isfree': False, 'init': 8.0, 'units': 'Msun',
                           'depends_on': transforms.nzsfh_to_logmass}

_beta_nzsfh_['logzsol'] = {'N': 1, 'isfree': False, 'init': -0.5, 'units': r'$\log (Z/Z_\odot)$',
                           'depends_on': transforms.nzsfh_to_logzsol}

# --- SFH ---
_beta_nzsfh_["sfh"] = {'N': 1, 'isfree': False, 'init': 3}

_beta_nzsfh_['logsfr_ratios'] = {'N': nbins_sfh-1, 'isfree': False, 'init': 0.0,
                                 'depends_on': transforms.nzsfh_to_logsfr_ratios}

_beta_nzsfh_["mass"] = {'N': nbins_sfh, 'isfree': False, 'init': 1e6, 'units': r'M$_\odot$',
                        'depends_on': transforms.logsfr_ratios_to_masses}

_beta_nzsfh_['agebins'] = {'N': nbins_sfh, 'isfree': False,
                           'init': transforms.zred_to_agebins_pbeta(np.atleast_1d(0.5), np.zeros(nbins_sfh)),
                           'depends_on': transforms.zred_to_agebins_pbeta}

TemplateLibrary["beta"] = (_beta_nzsfh_,
                           "The prospector-beta model; Wang, Leja, et al. 2023")
