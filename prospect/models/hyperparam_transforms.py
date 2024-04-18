#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hyperparam_transforms.py -- This module contains parameter transformations that are
used in the stochastic SFH prior.

These are taken from the implementation of
https://ui.adsabs.harvard.edu/abs/2024ApJ...961...53I/abstract given in
https://github.com/kartheikiyer/GP-SFH

They can be used as ``"depends_on"`` entries in parameter specifications.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM

__all__ = ["get_sfr_covar", "sfr_covar_to_sfr_ratio_covar"]


# --------------------------------------
# --- Functions/transforms for stochastic SFH prior ---
# --------------------------------------


# Creating a base class that simplifies a lot of things.
# The way this is set up, you can pass a kernel as an argument
# to compute the covariance matrix and draw samples from it.
class simple_GP_sfh():

    """
    A class that creates and holds information about a specific
    kernel, and can generate samples from it.

    From https://github.com/kartheikiyer/GP-SFH

    Attributes
    ----------
    tarr: fiducial time array used to draw samples
    kernel: accepts an input function as an argument,
        of the format:

            def kernel_function(delta_t, **kwargs):
                ... function interior ...
                return kernel_val[array of len(delta_t)]

    Methods
    -------
    get_covariance_matrix
        [although this has double for loops for maximum flexibility
        with generic kernel functions, it only has to be computed once,
        which makes drawing random samples super fast once it's computed.]
    sample_kernel
    plot_samples
    plot_kernel
    [to-do] condition on data

    """

    def __init__(self, sp = 'none', cosmo = FlatLambdaCDM(H0=70, Om0=0.3), zval = 0.1):


        self.kernel = []
        self.covariance_matrix = []
        self.zval = zval
        self.sp = sp
        self.cosmo = cosmo
        self.get_t_univ()
        self.get_tarr()


    def get_t_univ(self):

        self.t_univ = self.cosmo.age(self.zval).value
        return

    def get_tarr(self, n_tarr = 1000):

        self.get_t_univ()
        if n_tarr > 1:
            self.tarr = np.linspace(0,self.t_univ, n_tarr)
        elif n_tarr < 1:
            self.tarr = np.arange(0,self.t_univ, n_tarr)
        else:
            raise('Undefined n_tarr: expected int or float.')
        return


    def get_covariance_matrix(self, show_prog = True, **kwargs):
        """
        Evaluate covariance matrix with a particular kernel
        """

        cov_matrix = np.zeros((len(self.tarr),len(self.tarr)))

        if show_prog == True:
            iterrange = range(len(cov_matrix))
        else:
            iterrange = range(len(cov_matrix))
        for i in iterrange:
            for j in range(len(cov_matrix)):
                    cov_matrix[i,j] = self.kernel(self.tarr[i] - self.tarr[j], **kwargs)

        return cov_matrix



def extended_regulator_model_kernel_paramlist(delta_t, kernel_params, base_e_to_10 = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    kernel_params = [sigma, tau_eq, tau_in, sigma_gmc, tau_gmc]
    sigma: \sigma, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

    from https://github.com/kartheikiyer/GP-SFH
    """

    sigma, tau_eq, tau_in, sigma_gmc, tau_gmc = kernel_params

    if base_e_to_10 == True:
        # in TCF20, this is defined in base e, so convert to base 10
        sigma = sigma*np.log10(np.e)
        sigma_gmc = sigma_gmc*np.log10(np.e)

    tau = np.abs(delta_t)

    if tau_in == tau_eq:
        c_reg = sigma**2 * (1 + tau/tau_eq) * (np.exp(-tau/tau_eq))
    else:
        c_reg = sigma**2 / (tau_in - tau_eq) * (tau_in*np.exp(-tau/tau_in) - tau_eq*np.exp(-tau/tau_eq))

    c_gmc = sigma_gmc**2 * np.exp(-tau/tau_gmc)

    kernel_val = (c_reg + c_gmc)
    return kernel_val



def get_sfr_covar(psd_params, agebins=[], **extras):
    """Caluclates SFR covariance matrix for a given set of PSD parameters and agebins
    PSD parameters must be in the order: [sigma_reg, tau_eq, tau_in, sigma_dyn, tau_dyn]

    from https://github.com/kartheikiyer/GP-SFH

    Returns
    -------
    covar_matrix: (Nbins, Nbins)-dim array of covariance values for SFR
    """

    bincenters = np.array([np.mean(agebins[i]) for i in range(len(agebins))])
    bincenters = (10**bincenters)/1e9
    case1 = simple_GP_sfh()
    case1.tarr = bincenters
    case1.kernel = extended_regulator_model_kernel_paramlist
    covar_matrix = case1.get_covariance_matrix(kernel_params = psd_params, show_prog=False)

    return covar_matrix


def sfr_covar_to_sfr_ratio_covar(covar_matrix):
    """Caluclates log SFR ratio covariance matrix from SFR covariance matrix

    from https://github.com/kartheikiyer/GP-SFH

    Returns
    -------
    sfr_ratio_covar: (Nbins-1, Nbins-1)-dim array of covariance values for log SFR
    """

    dim = covar_matrix.shape[0]

    sfr_ratio_covar = []

    for i in range(dim-1):
        row = []
        for j in range(dim-1):
            cov = covar_matrix[i][j] - covar_matrix[i+1][j] - covar_matrix[i][j+1] + covar_matrix[i+1][j+1]
            row.append(cov)
        sfr_ratio_covar.append(row)

    return np.array(sfr_ratio_covar)