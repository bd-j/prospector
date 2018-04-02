# ---------
# A set of predefined "base" prospector model specifications that can be used
# and then altered.
# ---------

from copy import deepcopy
import numpy as np
from . import priors
from . import transforms

__all__ = ["_basic_", "_dust_emission_", "_nebular_", "_agn_",
           "_ssp_", "parametric_", "_nonpar_",
           "_alpha_",
#           "spectroscopy", "specphot"
           ]


# A template for what parameter configuration element should look like
par_name = {"N": 1,
            "isfree": True,
            "init": 0.5, "units": "",
            "prior": priors.TopHat(mini=0, maxi=1.0),
            "depends_on": None}


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

_basic_ = {"zred":zred,
           "mass": mass,
           "logzsol": logzsol,
           "dust2": dust2}

tage = {"N": 1, "isfree": True,
        "init": 1, "units": "Gyr",
        "prior": priors.TopHat(mini=0.001, maxi=13.8)}


# --------------------------
# ---  Dust emission ----
# --------------------------

# FSPS / Draine & Li parameter
duste_umin  = {"N": 1, "isfree": False,
               "init": 1.0, "units": 'MMP83 local MW intensity',
               "prior": priors.TopHat(mini=0.1, maxi=25)}

# FSPS / Draine & Li parameter
duste_qpah  = {"N": 1, "isfree": False,
               'init': 4.0, "units": 'Percent mass fraction of PAHs in dust.',
               "prior": priors.TopHat(mini=0.5, maxi=7.0)}

# FSPS / Draine & Li parameter
duste_gamma = {"N": 1, "isfree": False,
               "init": 0.0, "units": 'Mass fraction of dust in high radiation intensity.',
               "prior": priors.LogUniform(mini=1e-3, maxi=0.15)}

_dust_emission_ = {"duste_umin": duste_umin,
                   "duste_qpah": duste_qpah,
                   "duste_gamma": duste_gamma}

# --------------------------
# --- Nebular Emission ---
# --------------------------
add_neb = {'N': 1, 'isfree': False, 'init': True}

# FSPS parameter.  See Byler et al 2017
# Note this depends on stellar metallicity
gas_logz = {'N': 1, 'isfree': False,
            'init': 0.0, 'units': r'log Z/Z_\odot',
            'depends_on': transforms.stellar_logzsol,
            'prior': priors.TopHat(mini=-2.0, maxi=0.5)}

# FSPS parameter.  See Byler et al 2017
gas_logu = {"N": 1, 'isfree': False,
            'init': -2.0, 'units': r"Q_H/N_H",
            'prior': priors.TopHat(mini=-4, maxi=-1)}

_nebular_ = {"add_nebular_emission": add_neb,
             "gas_logz": gas_logz,
             "gas_logu": gas_logu}


# --------------------------
# --- AGN Torus Emission ---
# --------------------------
# FSPS parameter.
fagn = {'N': 1, 'isfree': False,
        'init': -2.0, 'units': r'L_{AGN}/L_*',
        'prior': priors.LogUniform(mini=-5.0, maxi=0.1)}

# FSPS parameter.  See Byler et al 2017
agn_tau = {"N": 1, 'isfree': False,
           "init": 1.0, 'units': r"optical depth",
            'prior': priors.LogUniform(mini=5.0, maxi=150.)}

_agn_ = {"fagn": fagn,
         "agn_tau": agn_tau}

    
# ----------------------------
# --- Sspecific models ---
# ----------------------------

# --- SSP (i.e. clusters) ---
_ssp_ = deepcopy(_basic_)
_ssp_["sfh"] = {"N": 1, "isfree": False, "init": 0, "units": "FSPS index"}
_ssp_["tage"] = tage

# --- Parametric SFH with photometry -----
_parameteric_ = deepcopy(_basic_)
_parameteric_["sfh"]  = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
_parameteric_["tage"] = tage
_parameteric_["tau"]  = {"N": 1, "isfree": True,
                         "init": 1, "units": "Gyr^{-1}",
                         "prior": priors.LogUniform(mini=0.1, maxi=30)}

# --- Non-parametric with photometry
# Using the dirichlet prior on SFR fractions.

_nonpar_ = deepcopy(_basic_)

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

def adjust_nonpar(parset, agelims=[0., 8., 9., 10.]):
    pass


# --- Prospector-alpha ---
_alpha_ = deepcopy(_nonpar_)
_alpha_.update(deepcopy(_dust_emission_))
_alpha_.update(deepcopy(_nebular_))
_alpha_.update(deepcopy(_agn_))

# Set the dust emission free
_alpha_["duste_qpah"]["isfree"] = True
_alpha_["duste_umin"]["isfree"] = True
_alpha_["duste_gamma"]["isfree"] = True

# Complexify the attenuation
_alpha_["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
_alpha_["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
_alpha_["dust1"]      = {"N": 1, "isfree": True,
                         "init": 0.0, "units": "optical depth towards young stars",
                         "prior": priors.TopHat(mini=0.0, maxi=4.0)}
_alpha_["dust_index"] = {"N": 1, "isfree": True,
                         "init": 0.0, "units": "power-law multiplication of Calzetti",
                         "prior": priors.TopHat(mini=-2.0, maxi=0.5)}

