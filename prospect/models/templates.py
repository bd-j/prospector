# ---------
# A set of predefined "base" prospector model specifications that can be used
# and then altered.
# ---------

from copy import deepcopy
import numpy as np
from . import priors
from . import transforms

__all__ = ["TemplateLibrary",
           "describe",
           "adjust_nonpar_bins",
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


def describe(parset):
    ttext = "Free Parameters: (name: prior) \n-----------\n"
    free = ["{}: {}".format(k, v["prior"])
            for k, v in list(parset.items()) if v["isfree"]]
    ttext += "  " + "\n  ".join(free)

    ftext = "Fixed Parameters: (name: value [, depends_on]) \n-----------\n"
    fixed = ["{}: {} {}".format(k, v["init"], v.get("depends_on", ""))
            for k, v in list(parset.items()) if not v["isfree"]]
    ftext += "  " + "\n  ".join(fixed)
    return ttext + "\n\n" + ftext


def adjust_nonpar_bins(parset, agelims=[0., 8., 9., 10.]):
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


TemplateLibrary = Directory()

# A template for what parameter configuration element should look like
par_name = {"N": 1,
            "isfree": True,
            "init": 0.5,
            "prior": priors.TopHat(mini=0, maxi=1.0),
            "depends_on": None,
            "units": "",}

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

_basic_ = {"zred":zred,
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
_parametric_["tau"]  = {"N": 1, "isfree": True,
                        "init": 1, "units": "Gyr^{-1}",
                        "prior": priors.LogUniform(mini=0.1, maxi=30)}

TemplateLibrary["parametric_sfh"] = (_parametric_,
                                     ("Basic set of (free) parameters for a delay-tau SFH."))


# --------------------------
# ---  Dust emission ----
# --------------------------

duste_umin  = {"N": 1, "isfree": False,
               "init": 1.0, "units": 'MMP83 local MW intensity',
               "prior": priors.TopHat(mini=0.1, maxi=25)}

duste_qpah  = {"N": 1, "isfree": False,
               'init': 4.0, "units": 'Percent mass fraction of PAHs in dust.',
               "prior": priors.TopHat(mini=0.5, maxi=7.0)}


duste_gamma = {"N": 1, "isfree": False,
               "init": 0.0, "units": 'Mass fraction of dust in high radiation intensity.',
               "prior": priors.LogUniform(mini=1e-3, maxi=0.15)}

_dust_emission_ = {"duste_umin": duste_umin,    # FSPS / Draine & Li parameter
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

# --------------------------
# --- AGN Torus Emission ---
# --------------------------

fagn = {'N': 1, 'isfree': False,
        'init': -2.0, 'units': r'L_{AGN}/L_*',
        'prior': priors.LogUniform(mini=-5.0, maxi=0.1)}

agn_tau = {"N": 1, 'isfree': False,
           "init": 1.0, 'units': r"optical depth",
           'prior': priors.LogUniform(mini=5.0, maxi=150.)}

_agn_ = {"fagn": fagn,       # FSPS parameter.
         "agn_tau": agn_tau  # FSPS parameter.
         }

TemplateLibrary["agn"] = (_agn_,
                          ("The set of (fixed) AGN dusty torus emission parameters."))

# --------------------------
# --- IGM Absorption ---
# --------------------------
add_igm = {'N': 1, 'isfree': False, 'init': True}

igm_fact ={'N': 1, 'isfree': False, 'init': 1.0,
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
               "sigma_smooth": sigma_smooth # prospecter `smoothspec` parameter
               }

TemplateLibrary["spectral_smoothing"] = (_smoothing_,
                                        ("Set of parameters for spectal smoothing."))


# --------------------------
# --- Spectral calibration
# -------------------------

spec_norm = {'N': 10, 'isfree': False,
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
# --- SF Bursts ---
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

_nonpar_lm_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  Its length needs to be modified based on the number of bins
_nonpar_lm_["mass"]       = {'N': 3, 'isfree': True, 'init': 1e6, 'units': r'M$_\odot$',
                          'prior': priors.LogUniform(mini=1e5, maxi=1e12)}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_nonpar_lm_["agebins"]    = {'N': 3, 'isfree': False,
                          'init': [[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]],
                          'units': 'log(yr)'}
# This is the *total* stellar mass formed
_nonpar_lm_["total_mass"] = {"N": 1, "isfree": False, "init": 1e10, "units": "Solar masses formed",
                             "depends_on": transforms.total_mass}

TemplateLibrary["nonpar_logm_sfh"] = (_nonpar_lm_,
                                    "Non-parameteric SFH fitting for mass in fixed time bins")

# ----------------------------
# --- Continuity SFH ----
# ----------------------------
# A non-parametric SFH model of mass in fixed time bins with a smoothness prior

_nonpar_continuity_ = TemplateLibrary["ssp"]

_nonpar_continuity_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
# This is the *total*  mass formed, as a variable
_nonpar_continuity_["logmass"] = {"N": 1, "isfree": True, "init": 10, 'units': 'Msun',
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
_nonpar_continuity_["logsfr_ratios"] = {'N': 2, 'isfree': True, 'init': [0.0,0.0],
                                       'prior':priors.StudentT(mean=np.full(2,0.0),
                                                               scale=np.full(2,0.3),
                                                               df=np.full(2,2))}
TemplateLibrary["nonpar_continuity_sfh"] = (_nonpar_continuity_,
                                            "Non-parameteric SFH fitting for mass in fixed time bins with a smoothness prior")

# ----------------------------
# --- Flexible Continuity SFH ----
# ----------------------------
# A non-parametric SFH model of mass in flexible time bins with a smoothness prior

_nonpar_continuity_flex_ = TemplateLibrary["ssp"]

_nonpar_continuity_flex_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
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
# parameters.  Its length needs to be modified based on the number of bins
_nonpar_continuity_flex_["mass"]       = {'N': 4, 'isfree': False, 'init': 1e6, 'units': r'M$_\odot$',
                                          'transform': transforms.logsfr_ratios_to_masses_flex}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_nonpar_continuity_flex_["agebins"]    = {'N': 4, 'isfree': False,
                                          'depends_on': transforms.logsfr_ratios_to_agebins,
                                          'init': [[0.0, 7.5], [7.5, 8.5],[8.5,9.7], [9.7, 10.0]],
                                          'units': 'log(yr)'}
TemplateLibrary["nonpar_continuity_flex_sfh"] = (_nonpar_continuity_flex_,
                                                 "Non-parameteric SFH fitting for mass in flexible time bins with a smoothness prior")
# ----------------------------
# --- Dirichlet SFH ----
# ----------------------------
# Using the dirichlet prior on SFR fractions in bins of constant SF.

_nonpar_ = TemplateLibrary["ssp"]

_nonpar_["sfh"]        = {"N": 1, "isfree": False, "init": 3, "units": "FSPS index"}
# This will be the mass in each bin.  It depends on other free and fixed
# parameters.  It's length needs to be modified based on the number of bins
_nonpar_["mass"]       = {'N': 3, 'isfree': False, 'init': 1., 'units': r'M$_\odot$',
                          'depends_on': transforms.zfrac_to_masses}
# This gives the start and stop of each age bin.  It can be adjusted and its
# length must match the lenth of "mass"
_nonpar_["agebins"]    = {'N': 3, 'isfree': False,
                          'init': [[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]],
                          'units': 'log(yr)'}
# Auxiliary variable used for sampling sfr_fractions from dirichlet. This
# *must* be adjusted depending on the number of bins
_nonpar_["z_fraction"] = {"N": 2, 'isfree': True, 'init': [0, 0], 'units': None,
                          'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)}
# This is the *total* stellar mass formed
_nonpar_["total_mass"] = mass

TemplateLibrary["dirichlet_sfh"] = (_nonpar_,
                                    "Non-parameteric SFH with Dirichlet prior")


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
_alpha_["dust1"]      = {"N": 1, "isfree": True,
                         "init": 0.0, "units": "optical depth towards young stars",
                         "prior": priors.TopHat(mini=0.0, maxi=4.0)}
_alpha_["dust_index"] = {"N": 1, "isfree": True,
                         "init": 0.0, "units": "power-law multiplication of Calzetti",
                         "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
# in Gyr
alpha_agelims = np.array([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
_alpha_ = adjust_nonpar_bins(_alpha_, agelims=(np.log10(alpha_agelims) + 9))

TemplateLibrary["alpha"] = (_alpha_,
                            "The prospector-alpha model, Leja et al. 2017")
