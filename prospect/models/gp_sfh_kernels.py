# kernels corresponding to the various models used in the paper
# 1. White kernel: white_kernel
# 2. Damped RW: damped_random_walk_kernel
# 3. Regulator model: regulator_model_kernel
# 4b. extended_regulator_model_kernel
# 4b. extended_regulator_model_kernel_old
# 4c. extended_regulator_model_kernel_paramlist_old

import numpy as np

# ---------------- suggested model kernel values -------------------------------

kernel_params_MW_1dex = [1.0, 2500/1e3, 150/1e3, 0.03, 25/1e3]
kernel_params_dwarf_1dex = [1.0, 30/1e3, 150/1e3, 0.03, 10/1e3]
kernel_params_noon_1dex = [1.0, 200/1e3, 100/1e3, 0.03, 50/1e3]
kernel_params_highz_1dex = [1.0, 15/1e3, 16/1e3, 0.03, 6/1e3]

def convert_sigma_obs_to_ExReg(sigma_target, sigma_GMC_to_reg_ratio = 0.03):
    return np.sqrt(sigma_target**2/(1 + (sigma_GMC_to_reg_ratio)**2)), sigma_GMC_to_reg_ratio*np.sqrt(sigma_target**2/(1 + (sigma_GMC_to_reg_ratio)**2))

TCF20_scattervals = [0.17, 0.53, 0.24, 0.27]
TCF20_GMC_to_reg_ratio = 0.03

temp_sigma, temp_sigma_GMC = convert_sigma_obs_to_ExReg(TCF20_scattervals[0], TCF20_GMC_to_reg_ratio)
kernel_params_MW_TCF20 = [temp_sigma, 2500/1e3, 150/1e3, temp_sigma_GMC, 25/1e3]
temp_sigma, temp_sigma_GMC = convert_sigma_obs_to_ExReg(TCF20_scattervals[1], TCF20_GMC_to_reg_ratio)
kernel_params_dwarf_TCF20 = [temp_sigma, 30/1e3, 150/1e3, temp_sigma_GMC, 10/1e3]
temp_sigma, temp_sigma_GMC = convert_sigma_obs_to_ExReg(TCF20_scattervals[2], TCF20_GMC_to_reg_ratio)
kernel_params_noon_TCF20 = [temp_sigma, 200/1e3, 100/1e3, temp_sigma_GMC, 50/1e3]
temp_sigma, temp_sigma_GMC = convert_sigma_obs_to_ExReg(TCF20_scattervals[3], TCF20_GMC_to_reg_ratio)
kernel_params_highz_TCF20 = [temp_sigma, 15/1e3, 16/1e3, temp_sigma_GMC, 6/1e3]

# --------------------- kernels -------------------------------------

def white_kernel(delta_t, sigma=1.0, base_e_to_10 = False):
    """
    A basic implementation of a white noise kernel, with one parameter:
    A: sigma (not a real parameter in this case)
    """
    if base_e_to_10 == True:
        sigma = sigma*np.log10(np.e)

    kernel_val = np.zeros_like(delta_t)
    kernel_val[delta_t == 0] = sigma**2
    return kernel_val

def damped_random_walk_kernel(delta_t, sigma=1.0, tau_eq = 1.0, base_e_to_10 = False):
    """
    A basic implementation of a damped random walk kernel, with two parameters:
    sigma: \sigma, the amount of overall variance
    tau_eq: equilibrium timescale

    """
    if base_e_to_10 == True:
        sigma = sigma*np.log10(np.e)

    tau = np.abs(delta_t)
    kernel_val = (sigma**2) * np.exp(- tau / tau_eq)
    return kernel_val

def regulator_model_kernel(delta_t, sigma=1.0, tau_eq = 1.0, tau_in = 0.5, base_e_to_10 = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    sigma: \sigma, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

    """

    # in TCF20, this is defined in base e, so convert to base 10
    if base_e_to_10 == True:
        sigma = sigma*np.log10(np.e)

    tau = np.abs(delta_t)

    if tau_in == tau_eq:
        c_reg = sigma**2 * (1 + tau/tau_eq) * (np.exp(-tau/tau_eq))
    else:
        c_reg = sigma**2 / (tau_in - tau_eq) * (tau_in*np.exp(-tau/tau_in) - tau_eq*np.exp(-tau/tau_eq))

    # if tau_in == tau_eq:
    #     c_reg = sigma**2 * (1+ np.abs(delta_t)/tau_eq) * (np.exp(-np.abs(delta_t)/tau_eq)/(2*tau_eq))
    # else:
    #     c_reg = sigma**2 / (tau_in**2 - tau_eq**2) * (tau_in * np.exp(-np.abs(delta_t) / tau_in) - tau_eq * np.exp(-np.abs(delta_t) / tau_eq))

    return c_reg

def extended_regulator_model_kernel(delta_t, sigma=1.0, tau_eq = 1.0, tau_in = 0.5, sigma_gmc = 0.01, tau_gmc = 0.001, base_e_to_10 = False, return_components = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    sigma: \sigma_{gas}, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

    """

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

    if return_components == True:
        return kernel_val, c_reg, c_gmc
    else:
        return kernel_val

def extended_regulator_model_kernel_paramlist(delta_t, kernel_params, base_e_to_10 = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    kernel_params = [sigma, tau_eq, tau_in, sigma_gmc, tau_gmc]
    sigma: \sigma, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

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

def extended_regulator_model_PSD(f, sigma=1.0, tau_eq = 1.0, tau_in = 0.5, sigma_gmc = 0.01, tau_gmc = 0.001, base_e_to_10 = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    sigma: \sigma, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

    """

    if base_e_to_10 == True:
        # in TCF20, this is defined in base e, so convert to base 10
        sigma_base10 = sigma*np.log10(np.e)
        sigma_gmc_base10 = sigma_gmc*np.log10(np.e)
    else:
        sigma_base10 = sigma
        sigma_gmc_base10 = sigma_gmc

    tpi = 2*np.pi

    psd_reg = sigma_base10**2 / (1 + ((tpi*tau_eq*1e3)**2 + (tpi*tau_in*1e3)**2)*(f**2) + ((tpi*tau_eq*1e3)**2 * (tpi*tau_in*1e3)**2)*(f**4))
    psd_gmc = sigma_gmc_base10**2 / (1 + (tpi*tau_gmc*1e3*f)**2)
    psd_val = psd_reg + psd_gmc

    return psd_val, psd_reg, psd_gmc

# ---------------------------------------------------------------------
# ------------------------ deprecated ---------------------------------
# ---------------------------------------------------------------------

def extended_regulator_model_kernel_old(delta_t, sigma=1.0, tau_eq = 1.0, tau_in = 0.5, sigma_gmc = 0.01, tau_gmc = 0.001, base_e_to_10 = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    sigma: \sigma_{gas}, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

    """

    if base_e_to_10 == True:
        # in TCF20, this is defined in base e, so convert to base 10
        sigma = sigma*np.log10(np.e)
        sigma_gmc = sigma_gmc*np.log10(np.e)

    if tau_in == tau_eq:
        c_reg = sigma**2 * (1+ np.abs(delta_t)/tau_eq) * (np.exp(-np.abs(delta_t)/tau_eq)/(2*tau_eq))
    else:
        c_reg = sigma**2 / (tau_in**2 - tau_eq**2) * (tau_in * np.exp(-np.abs(delta_t) / tau_in) - tau_eq * np.exp(-np.abs(delta_t) / tau_eq))
    c_gmc = sigma_gmc**2 / (2*tau_gmc) * np.exp(-np.abs(delta_t)/tau_gmc)
    kernel_val = (c_reg + c_gmc)
    return kernel_val

def extended_regulator_model_kernel_paramlist_old(delta_t, kernel_params, base_e_to_10 = False):
    """
    A basic implementation of the regulator model kernel, with five parameters:
    kernel_params = [sigma, tau_eq, tau_in, sigma_gmc, tau_gmc]
    sigma: \sigma, the amount of overall variance
    tau_eq: equilibrium timescale
    tau_x: inflow correlation timescale (includes 2pi factor)
    sigma_gmc: gmc variability
    tau_l: cloud lifetime

    """

    sigma, tau_eq, tau_in, sigma_gmc, tau_gmc = kernel_params

    if base_e_to_10 == True:
        # in TCF20, this is defined in base e, so convert to base 10
        sigma = sigma*np.log10(np.e)
        sigma_gmc = sigma_gmc*np.log10(np.e)

    if tau_in == tau_eq:
        c_reg = sigma**2 * (1+ np.abs(delta_t)/tau_eq) * (np.exp(-np.abs(delta_t)/tau_eq)/(2*tau_eq))
    else:
        c_reg = sigma**2 / (tau_in**2 - tau_eq**2) * (tau_in * np.exp(-np.abs(delta_t) / tau_in) - tau_eq * np.exp(-np.abs(delta_t) / tau_eq))
    c_gmc = sigma_gmc**2 / (2*tau_gmc) * np.exp(-np.abs(delta_t)/tau_gmc)
    kernel_val = (c_reg + c_gmc)
    return kernel_val