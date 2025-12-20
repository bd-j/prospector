#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""priors_beta.py -- This module contains the prospector-beta priors.
Ref: Wang, Leja, et al., 2023, ApJL.

Specifically, this module includes the following priors --
1. PhiMet         : p(logM|z)p(Z*|logM), i.e., mass function + mass-met
2. ZredMassMet    : p(z)p(logM|z)p(Z*|logM), i.e., number density + mass function + mass-met
3. DymSFH         : p(Z*|logM) & SFH(M, z), i.e., mass-met + SFH
4. DymSFHfixZred  : same as above, but keeping zred fixed to a user-specified value, 'zred', during fitting
5. PhiSFH         : p(logM|z)p(Z*|logM) & SFH(M, z), i.e., mass function + mass-met + SFH
6. PhiSFHfixZred  : same as above, but keeping zred fixed to a user-specified value, 'zred', during fitting
7. NzSFH          : p(z)p(logM|z)p(Z*|logM) & SFH(M, z),
                    i.e., number density + mass function + mass-met + SFH;
                    this is the full set of prospector-beta priors.

When called these return the ln-prior-probability, and they can also be used to
construct prior transforms (for nested sampling) and can be sampled from.
"""

import numpy as np

try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
from scipy.interpolate import interp1d
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from scipy.stats import t
from . import priors
from .prior_data import (
    MASSMET,
    SPL_TL_SFRD,
    F_AGE_Z,
    WMAP9_AGE,
)

__all__ = [
    "PhiMet",
    "ZredMassMet",
    "DymSFH",
    "DymSFHfixZred",
    "PhiSFH",
    "PhiSFHfixZred",
    "NzSFH",
]

_, AGE_GRID = WMAP9_AGE


class BetaPrior(priors.Prior):
    """Base class for Prospector-Beta priors, implementing shared logic for
    redshift, mass, metallicity, and SFH priors.
    """

    prior_params = [
        "zred_mini",
        "zred_maxi",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "logsfr_ratio_mini",
        "logsfr_ratio_maxi",
        "logsfr_ratio_tscale",
        "nbins_sfh",
        "const_phi",
        "zred",
    ]

    def __init__(
        self,
        parnames=[],
        name="",
        z_prior="uniform",  # 'uniform', 'fixed', 'dynamic'
        mass_prior="uniform",  # 'uniform', 'mass_function'
        sfh_prior=False,  # True/False
        **kwargs,
    ):
        """
        Parameters
        ----------
        z_prior : str
            Type of redshift prior: 'uniform', 'fixed', or 'dynamic' (calculated from number density).
        mass_prior : str
            Type of mass prior: 'uniform' or 'mass_function'.
        sfh_prior : bool
            Whether to include the SFH prior.
        """
        if len(parnames) == 0:
            # Filter prior_params to only those relevant for the subclass if needed,
            # or just use the full list as the base implementation.
            # To be safe and compatible with existing instantiations, we use the provided kwargs
            # to populate params.
            parnames = self.prior_params

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}
        self.name = name
        self.update(**kwargs)

        self.z_prior_type = z_prior
        self.mass_prior_type = mass_prior
        self.sfh_prior_flag = sfh_prior

        # --- Mass Prior Helper Setup ---
        # Handle mass_mini depending on if its a scalar or a function
        # This is needed for dynamic Z prior AND mass function calculations
        if callable(self.params.get("mass_mini")):
            self.mass_min_func = self.params["mass_mini"]
        else:
            self.mass_min_func = lambda z: self.params["mass_mini"]

        # --- Pre-calculate Mass Function Normalization ---
        # We need this for __call__ normalization and for dynamic redshift prior
        if self.mass_prior_type == "mass_function":
            self._setup_mass_normalization()

        # --- Redshift Prior Setup ---
        if self.z_prior_type == "fixed":
            self.zred = self.params.get("zred", 0.0)  # Ensure zred is available
        elif self.z_prior_type == "uniform":
            self.zred_dist = priors.FastUniform(
                a=self.params["zred_mini"], b=self.params["zred_maxi"]
            )
        elif self.z_prior_type == "dynamic":
            self._setup_dynamic_z_prior_from_norm()
        else:
            raise ValueError(f"Unknown z_prior type: {self.z_prior_type}")

        # --- Mass Prior Setup ---
        # Setup mass grid/dist
        if self.mass_prior_type == "uniform":
            self.mass_dist = priors.FastUniform(
                a=self.params["mass_mini"], b=self.params["mass_maxi"]
            )
        elif self.mass_prior_type == "mass_function":
            # If mass_mini is callable (NzSFH), we can't create a static global grid.
            if not callable(self.params.get("mass_mini")):
                self.mgrid = np.linspace(
                    self.params.get("mass_mini", 9.0), self.params["mass_maxi"], 101
                )
            else:
                self.mgrid = None

        # --- SFH Prior Setup ---
        if self.sfh_prior_flag:
            self.logsfr_ratios_dist = priors.FastTruncatedEvenStudentTFreeDeg2(
                hw=self.params["logsfr_ratio_maxi"],
                sig=self.params["logsfr_ratio_tscale"],
            )

    def _setup_dynamic_z_prior_from_norm(self):
        """Constructs p(z) ~ N(z) * dV/dz using pre-calculated N(z)."""

        dvol = cosmo.differential_comoving_volume(self._z_grid_norm).value
        pdf_z_unnorm = self._n_gal_z * dvol

        self.finterp_z_pdf, self.finterp_cdf_z = norm_pz(
            self.params["zred_mini"],
            self.params["zred_maxi"],
            self._z_grid_norm,
            pdf_z_unnorm,
        )

    def _setup_mass_normalization(self):
        """Calculates the integral of the mass function over the mass limits
        as a function of redshift. Used for normalization in __call__ and
        for the dynamic redshift prior."""

        # Define grid (high res for accuracy)
        self._z_grid_norm = np.linspace(
            self.params.get("zred_mini", 0), self.params.get("zred_maxi", 10), 4000
        )

        n_gal_z = []
        mass_max = self.params["mass_maxi"]

        for z in self._z_grid_norm:
            m_min = self.mass_min_func(z)
            if m_min >= mass_max:
                n_gal_z.append(0.0)
                continue

            m_integ_grid = np.linspace(m_min, mass_max, 100)
            phi = mass_func_at_z(
                z,
                m_integ_grid,
                const_phi=self.params["const_phi"],
                bounds=[m_min, mass_max],
            )
            n = simpson(y=phi, x=m_integ_grid)
            n_gal_z.append(n)

        self._n_gal_z = np.array(n_gal_z)

        # Create interpolator for fast normalization in __call__
        self.finterp_mass_norm = interp1d(
            self._z_grid_norm, self._n_gal_z, bounds_error=False, fill_value=0.0
        )

    def __len__(self):
        if self.sfh_prior_flag:
            return self.params["nbins_sfh"] + 2
        else:
            return 3

    @property
    def range(self):
        r = []
        # Z range
        if self.z_prior_type == "fixed":
            pass
        else:
            r.append((self.params["zred_mini"], self.params["zred_maxi"]))

        r.append((self.params.get("mass_mini", 9.0), self.params["mass_maxi"]))
        r.append((self.params["z_mini"], self.params["z_maxi"]))

        if self.sfh_prior_flag:
            r.append(
                (self.params["logsfr_ratio_mini"], self.params["logsfr_ratio_maxi"])
            )

        return tuple(r)

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)

        if x.ndim == 1:
            x = x[np.newaxis, :]
            scalar_input = True
        else:
            scalar_input = False

        zreds = x[..., 0]
        logms = x[..., 1]
        logzsols = x[..., 2]

        lnp = np.zeros_like(x)  # Shape (N, N_params)

        # --- 1. Redshift Prior ---
        if self.z_prior_type == "fixed":
            lnp[..., 0] = 0.0
        elif self.z_prior_type == "uniform":
            lnp[..., 0] = self.zred_dist(zreds)
        elif self.z_prior_type == "dynamic":
            # vectorizing finterp_z_pdf
            # finterp_z_pdf is from interp1d, which should handle vectors
            lnp[..., 0] = np.log(self.finterp_z_pdf(zreds))

        # --- 2. Mass Prior ---
        if self.mass_prior_type == "uniform":
            lnp[..., 1] = self.mass_dist(logms)
        elif self.mass_prior_type == "mass_function":
            p_mass = np.zeros_like(logms)
            for i in range(len(zreds)):
                z = zreds[i]
                m = logms[i]

                # Get bounds for this z
                m_min = self.mass_min_func(z)
                m_max = self.params["mass_maxi"]

                # Raw value
                phi_val = mass_func_at_z(
                    z, m, self.params["const_phi"], bounds=[m_min, m_max]
                )

                norm = self.finterp_mass_norm(z)

                if norm > 0:
                    p_mass[i] = phi_val / norm
                else:
                    p_mass[i] = 0.0

            # Suppress log(0) warnings
            with np.errstate(divide="ignore"):
                lnp[..., 1] = np.log(p_mass)

        # --- 3. Metallicity Prior ---
        # p(zsol) depends on mass
        met_dists = [
            priors.FastTruncatedNormal(
                a=self.params["z_mini"],
                b=self.params["z_maxi"],
                mu=loc_massmet(m),
                sig=scale_massmet(m),
            )
            for m in logms
        ]
        lnp[..., 2] = [dist(zsol) for dist, zsol in zip(met_dists, logzsols)]

        # --- 4. SFH Prior ---
        if self.sfh_prior_flag:
            logsfr_input = x[..., 3:]
            # logsfr_ratios depends on z and m

            p_sfh = []
            for i in range(len(zreds)):
                z = zreds[i]
                m = logms[i]
                current_sfh_input = logsfr_input[i]

                logsfr_ratios = expe_logsfr_ratios(
                    this_z=z,
                    this_m=m,
                    nbins_sfh=self.params["nbins_sfh"],
                    logsfr_ratio_mini=self.params["logsfr_ratio_mini"],
                    logsfr_ratio_maxi=self.params["logsfr_ratio_maxi"],
                )

                p_sfh_i = t.pdf(
                    current_sfh_input,
                    df=2,
                    loc=logsfr_ratios,
                    scale=self.params["logsfr_ratio_tscale"],
                )
                p_sfh.append(p_sfh_i)

            p_sfh = np.array(p_sfh)
            with np.errstate(divide="ignore"):
                lnp[..., 3:] = np.log(p_sfh)

        if scalar_input:
            return lnp[0]
        else:
            return lnp

    def sample(self, nsample=None, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)

        n = 1 if nsample is None else nsample

        # 1. Redshift
        if self.z_prior_type == "fixed":
            zred = np.full(n, self.zred)
        elif self.z_prior_type == "uniform":
            u_z = np.random.uniform(0, 1, size=n)
            zred = self.zred_dist.unit_transform(u_z)
        elif self.z_prior_type == "dynamic":
            u_z = np.random.uniform(0, 1, size=n)
            zred = self.finterp_cdf_z(u_z)

        # 2. Mass
        mass = np.zeros(n)
        if self.mass_prior_type == "uniform":
            # self.mass_dist is FastUniform
            u_m = np.random.uniform(0, 1, size=n)
            mass = self.mass_dist.unit_transform(u_m)
        elif self.mass_prior_type == "mass_function":
            for i, z in enumerate(zred):
                m_min = self.mass_min_func(z)
                m_max = self.params["mass_maxi"]

                # We need a grid for this z
                m_subgrid = np.linspace(m_min, m_max, 100)

                cdf_mass = cdf_mass_func_at_z(
                    z=z,
                    logm=m_subgrid,
                    const_phi=self.params["const_phi"],
                    bounds=[m_min, m_max],
                )

                mass[i] = draw_sample(xs=m_subgrid, cdf=cdf_mass)

        # 3. Metallicity
        met = np.zeros(n)
        for i, m in enumerate(mass):
            met_dist = priors.FastTruncatedNormal(
                a=self.params["z_mini"],
                b=self.params["z_maxi"],
                mu=loc_massmet(m),
                sig=scale_massmet(m),
            )
            met[i] = met_dist.sample()

        # 4. SFH
        if self.sfh_prior_flag:
            sfh_rvs = []
            for i in range(n):
                z = zred[i]
                m = mass[i]
                logsfr_ratios = expe_logsfr_ratios(
                    this_z=z,
                    this_m=m,
                    nbins_sfh=self.params["nbins_sfh"],
                    logsfr_ratio_mini=self.params["logsfr_ratio_mini"],
                    logsfr_ratio_maxi=self.params["logsfr_ratio_maxi"],
                )
                # t.rvs
                rvs = t.rvs(
                    df=2, loc=logsfr_ratios, scale=self.params["logsfr_ratio_tscale"]
                )
                rvs = np.clip(
                    rvs,
                    a_min=self.params["logsfr_ratio_mini"],
                    a_max=self.params["logsfr_ratio_maxi"],
                )
                sfh_rvs.append(rvs)
            sfh_rvs = np.array(sfh_rvs)  # Shape (n, nbins_sfh-1)

        # Return
        if nsample is None:
            res = [
                np.atleast_1d(zred[0]),
                np.atleast_1d(mass[0]),
                np.atleast_1d(met[0]),
            ]
            if self.sfh_prior_flag:
                res.append(np.atleast_1d(sfh_rvs[0]))
            return np.concatenate(res)
        else:
            # vstack
            res = [zred, mass, met]
            if self.sfh_prior_flag:
                # sfh_rvs is (n, bins) -> transpose to (bins, n) for vstack?
                res.append(sfh_rvs.T)
            return np.vstack(res)

    def unit_transform(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)

        # x is 1D array of length N_params (unit cube coordinates)

        # 1. Redshift
        zred_val = 0.0
        if self.z_prior_type == "fixed":
            zred_val = self.zred * 1
        elif self.z_prior_type == "uniform":
            zred_val = self.zred_dist.unit_transform(x[0])
        elif self.z_prior_type == "dynamic":
            zred_val = self.finterp_cdf_z(x[0])

        # 2. Mass
        mass_val = 0.0
        if self.mass_prior_type == "uniform":
            mass_val = self.mass_dist.unit_transform(x[1])
        elif self.mass_prior_type == "mass_function":
            m_min = self.mass_min_func(zred_val)
            m_max = self.params["mass_maxi"]
            m_subgrid = np.linspace(m_min, m_max, 100)

            cdf_mass = cdf_mass_func_at_z(
                z=zred_val,
                logm=m_subgrid,
                const_phi=self.params["const_phi"],
                bounds=[m_min, m_max],
            )
            mass_val = ppf(x[1], m_subgrid, cdf=cdf_mass)

        # 3. Metallicity
        met_dist = priors.FastTruncatedNormal(
            a=self.params["z_mini"],
            b=self.params["z_maxi"],
            mu=loc_massmet(mass_val),
            sig=scale_massmet(mass_val),
        )
        met_val = met_dist.unit_transform(x[2])

        res = [np.atleast_1d(zred_val), np.atleast_1d(mass_val), np.atleast_1d(met_val)]

        # 4. SFH
        if self.sfh_prior_flag:
            logsfr_ratios = expe_logsfr_ratios(
                this_z=zred_val,
                this_m=mass_val,
                nbins_sfh=self.params["nbins_sfh"],
                logsfr_ratio_mini=self.params["logsfr_ratio_mini"],
                logsfr_ratio_maxi=self.params["logsfr_ratio_maxi"],
            )

            # Need to transform each bin
            # x[3:] corresponds to sfh bins
            sfh_unit = x[3:]
            sfh_vals = np.zeros_like(logsfr_ratios)

            for i in range(len(sfh_vals)):
                sfh_vals[i] = (
                    self.logsfr_ratios_dist.unit_transform(sfh_unit[i])
                    + logsfr_ratios[i]
                )

            sfh_vals = np.clip(
                sfh_vals,
                a_min=self.params["logsfr_ratio_mini"],
                a_max=self.params["logsfr_ratio_maxi"],
            )
            res.append(np.atleast_1d(sfh_vals))

        return np.concatenate(res)


# --- Re-implementing existing classes using BetaPrior ---


class PhiMet(BetaPrior):
    """p(logM|z)p(Z*|logM)
    Mass function + mass-met
    Z prior: Uniform
    Mass prior: Mass Function
    SFH: No
    """

    prior_params = [
        "zred_mini",
        "zred_maxi",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="uniform",
            mass_prior="mass_function",
            sfh_prior=False,
            **kwargs,
        )


class ZredMassMet(BetaPrior):
    """p(z)p(logM|z)p(Z*|logM)
    Number density + mass function + mass-met
    Z prior: Dynamic (Number Density)
    Mass prior: Mass Function
    SFH: No
    """

    prior_params = [
        "zred_mini",
        "zred_maxi",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="dynamic",
            mass_prior="mass_function",
            sfh_prior=False,
            **kwargs,
        )


class DymSFH(BetaPrior):
    """p(Z*|logM) & SFH(M, z)
    Mass-met + SFH
    Z prior: Uniform
    Mass prior: Uniform
    SFH: Yes
    """

    prior_params = [
        "zred_mini",
        "zred_maxi",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "logsfr_ratio_mini",
        "logsfr_ratio_maxi",
        "logsfr_ratio_tscale",
        "nbins_sfh",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="uniform",
            mass_prior="uniform",
            sfh_prior=True,
            **kwargs,
        )


class DymSFHfixZred(BetaPrior):
    """Same as DymSFH, but fixed zred
    Z prior: Fixed
    Mass prior: Uniform
    SFH: Yes
    """

    prior_params = [
        "zred",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "logsfr_ratio_mini",
        "logsfr_ratio_maxi",
        "logsfr_ratio_tscale",
        "nbins_sfh",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="fixed",
            mass_prior="uniform",
            sfh_prior=True,
            **kwargs,
        )


class PhiSFH(BetaPrior):
    """p(logM|z)p(Z*|logM) & SFH(M, z)
    Mass function + mass-met + SFH
    Z prior: Uniform
    Mass prior: Mass Function
    SFH: Yes
    """

    prior_params = [
        "zred_mini",
        "zred_maxi",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "logsfr_ratio_mini",
        "logsfr_ratio_maxi",
        "logsfr_ratio_tscale",
        "nbins_sfh",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="uniform",
            mass_prior="mass_function",
            sfh_prior=True,
            **kwargs,
        )


class PhiSFHfixZred(BetaPrior):
    """Same as PhiSFH, but fixed zred
    Z prior: Fixed
    Mass prior: Mass Function
    SFH: Yes
    """

    prior_params = [
        "zred",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "logsfr_ratio_mini",
        "logsfr_ratio_maxi",
        "logsfr_ratio_tscale",
        "nbins_sfh",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="fixed",
            mass_prior="mass_function",
            sfh_prior=True,
            **kwargs,
        )


class NzSFH(BetaPrior):
    """p(z)p(logM|z)p(Z*|logM) & SFH(M, z)
    Number density + mass function + mass-met + SFH
    Z prior: Dynamic
    Mass prior: Mass Function
    SFH: Yes
    """

    prior_params = [
        "zred_mini",
        "zred_maxi",
        "mass_mini",
        "mass_maxi",
        "z_mini",
        "z_maxi",
        "logsfr_ratio_mini",
        "logsfr_ratio_maxi",
        "logsfr_ratio_tscale",
        "nbins_sfh",
        "const_phi",
    ]  # mass is in log10

    def __init__(self, parnames=[], name="", **kwargs):
        super().__init__(
            parnames=parnames,
            name=name,
            z_prior="dynamic",
            mass_prior="mass_function",
            sfh_prior=True,
            **kwargs,
        )


############################# necessary functions #############################


def scale_massmet(mass):
    """std of the Gaussian approximating the mass-met relationship"""
    upper_84 = np.interp(mass, MASSMET[:, 0], MASSMET[:, 3])
    lower_16 = np.interp(mass, MASSMET[:, 0], MASSMET[:, 2])
    return upper_84 - lower_16


def loc_massmet(mass):
    """mean of the Gaussian approximating the mass-met relationship"""
    return np.interp(mass, MASSMET[:, 0], MASSMET[:, 1])


############ Mass function in Leja+20    ############
############ Code modified from appendix ############
def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """
    Generate a Schechter function (in dlogm).
    """
    phi = (
        (10**logphi)
        * np.log(10)
        * 10 ** ((logm - logmstar) * (alpha + 1))
        * np.exp(-(10 ** (logm - logmstar)))
    )
    return phi


def parameter_at_z0(y, z0, z1=0.2, z2=1.6, z3=3.0):
    """
    Compute parameter at redshift ‘z0‘ as a function
    of the polynomial parameters ‘y‘ and the
    redshift anchor points ‘z1‘, ‘z2‘, and ‘z3‘.
    """
    y1, y2, y3 = y
    a = ((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (
        z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)
    )
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c


# Continuity model median parameters + 1-sigma uncertainties.
pars = {
    "logphi1": [-2.44, -3.08, -4.14],
    "logphi1_err": [0.02, 0.03, 0.1],
    "logphi2": [-2.89, -3.29, -3.51],
    "logphi2_err": [0.04, 0.03, 0.03],
    "logmstar": [10.79, 10.88, 10.84],
    "logmstar_err": [0.02, 0.02, 0.04],
    "alpha1": [-0.28],
    "alpha1_err": [0.07],
    "alpha2": [-1.48],
    "alpha2_err": [0.1],
}


def draw_at_z(z0=1.0):
    """The Leja+20 mass function is only defined over 0.2<=z<=3.
    If 'z0' is outside the range, we use the z=0.2 and z=3 parameter values
    in this function.
    """
    if hasattr(z0, "__len__"):
        _z0 = np.array(z0)
        _z0[_z0 < 0.2] = 0.2
        _z0[_z0 > 3.0] = 3.0
    else:
        if z0 < 0.2:
            _z0 = 0.2
        elif z0 > 3.0:
            _z0 = 3.0
        else:
            _z0 = z0 * 1

    draws = {}

    for par in ["logphi1", "logphi2", "logmstar", "alpha1", "alpha2"]:
        samp = pars[par]
        if par in ["logphi1", "logphi2", "logmstar"]:
            draws[par] = parameter_at_z0(samp, _z0)
        else:
            draws[par] = np.array(samp)

    return draws


def low_z_mass_func(z0, logm):
    """Mass function in Leja+20
    logm: an array of [mass_mini, ..., mass_maxi], or a float.
    returns: an array of phi as a function of logm, or phi at z0.
    """
    draws = draw_at_z(z0=z0)
    phi1 = schechter(
        logm,
        draws["logphi1"],  # primary component
        draws["logmstar"],
        draws["alpha1"],
    )
    phi2 = schechter(
        logm,
        draws["logphi2"],  # secondary component
        draws["logmstar"],
        draws["alpha2"],
    )

    phi = phi1 + phi2

    return np.squeeze(phi)


############ Mass function in Tacchella+18 ############
def scale(x, zrange=[3, 4], trigrange=[0, np.pi / 2]):
    x_std = (x - zrange[0]) / (zrange[1] - zrange[0])
    x_scaled = x_std * (trigrange[1] - trigrange[0]) + trigrange[0]

    return x_scaled


def cos_weight(z, zrange=[3, 4]):
    """cos^2 weighted average

    zrange: min and max of the redshift range over which we will take the cos^2 weighted average
    """

    z_scaled = scale(z, zrange=zrange)
    w_l20 = np.cos(z_scaled) ** 2
    w_t18 = 1 - np.cos(z_scaled) ** 2

    return np.array([w_l20, w_t18])


# Best-fit parameters from table 2.
z_t18 = np.arange(4, 13, 1)
phi_t18 = np.array([261.9, 201.2, 140.5, 78.0, 38.4, 37.3, 8.1, 3.9, 1.1])
phi_t18 *= 1e-5
logm_t18 = np.array([10.16, 9.89, 9.62, 9.38, 9.18, 8.74, 8.79, 8.50, 8.50])
m_t18 = 10**logm_t18
alpha_t18 = np.array([-1.54, -1.59, -1.64, -1.70, -1.76, -1.80, -1.92, -2.00, -2.10])


def schechter_t18(m, phi_star, m_star, alpha):
    """In linear space"""
    return phi_star * (m / m_star) ** (alpha + 1) * np.exp(-m / m_star)


def high_z_mass_func_discreate(z0, this_m):
    # no boundary check is done here
    idx = np.squeeze(np.where(z_t18 == int(np.round(z0))))
    return schechter_t18(
        m=this_m, phi_star=phi_t18[idx], m_star=m_t18[idx], alpha=alpha_t18[idx]
    )


def high_z_mass_func(z0, this_m):
    """Take cos^2 weighted average in each bin, with zrange being the bin edges."""
    if z0 <= 4.0:
        return high_z_mass_func_discreate(4.0, this_m)
    elif z0 >= 12.0:
        return high_z_mass_func_discreate(12.0, this_m)
    else:
        zrange = [int(z0), int(z0) + 1]

        phi0 = high_z_mass_func_discreate(zrange[0], this_m)
        phi1 = high_z_mass_func_discreate(zrange[1], this_m)

        w = cos_weight(z0, zrange=zrange)
        phi = phi0 * w[0] + phi1 * w[1]
        return phi


def mass_func_at_z(z, this_logm, const_phi=False, bounds=[6.0, 12.5]):
    """
    if const_phi == True: use mass functions in Leja+20 only;
                          no redshfit evolution outside the range 0.2 <= z <= 3.0;
        i.e.,
        z<=0.2: use mass function at z=0.2;
        0.2<=z<=3.0: defined in Leja+20;
        z>=3: use mass function at z=3.0.
    if const_phi == False: combine Leja+20 and Tacchella+18 mass functions;
        i.e.,
        z<=3: mass function in Leja+20; continuous in redshift.
        3<z<4: cos^2 weighted average of L20 mass function at z=3 and T18 mass function at z=4;
               i.e., we weight the L20 mass function as cos^2(0, pi/2), and T18 as 1 - cos^2(0, pi/2).
        4<=z<=12: T18 mass function; discreate in redshift by definition;
                  we take cos^2 weighted average in each bin, with zrange being the bin edges.
        z>12: T18 mass function at z=12.
    """
    if not hasattr(this_logm, "__len__"):
        if this_logm < bounds[0] or this_logm > bounds[1]:
            return np.zeros_like(this_logm)

    if const_phi:
        phi = low_z_mass_func(z, this_logm)
    else:
        if z <= 3.0:
            phi = low_z_mass_func(z0=z, logm=this_logm)
        elif z > 3.0 and z < 4.0:
            phi_lowz = low_z_mass_func(z0=3, logm=this_logm)
            phi_highz = high_z_mass_func(z0=4, this_m=10**this_logm)
            w = cos_weight(z)
            phi = phi_lowz * w[0] + phi_highz * w[1]
        elif z >= 4.0 and z <= 12.0:
            phi = high_z_mass_func(z0=z, this_m=10**this_logm)
        else:
            phi = high_z_mass_func(z0=12, this_m=10**this_logm)

    if hasattr(this_logm, "__len__"):
        phi[this_logm < bounds[0]] = 0
        phi[this_logm > bounds[1]] = 0
    return np.squeeze(phi)


############ Empirical PDF & CDF ############
def pdf_mass_func_at_z(z, logm, const_phi, bounds):
    phi_50 = mass_func_at_z(z, logm, const_phi, bounds)
    p_phi_int = simpson(y=phi_50, x=logm)
    pdf_at_m = phi_50 / p_phi_int
    return pdf_at_m


def cdf_mass_func_at_z(z, logm, const_phi, bounds):
    """
    logm: an array of [mass_mini, ..., mass_maxi], or a float
    """
    pdf_at_m = pdf_mass_func_at_z(z, logm=logm, const_phi=const_phi, bounds=bounds)
    cdf_of_m = np.cumsum(pdf_at_m)
    cdf_of_m /= max(cdf_of_m)

    # may have small numerical errors; force cdf to be within [0, 1]
    clean = np.where(cdf_of_m < 0)
    cdf_of_m[clean] = 0
    cdf_of_m[0] = 0
    cdf_of_m[-1] = 1

    return cdf_of_m


def ppf(x, xs, cdf):
    """Go from a value x of the CDF (between 0 and 1) to
    the corresponding parameter value.
    """
    func_interp = interp1d(cdf, xs, bounds_error=False, fill_value=0)
    param = func_interp(x)
    return param


def draw_sample(xs, cdf, nsample=None):
    """Draw sample(s) from any cdf"""
    u = np.random.uniform(0, 1, size=nsample)
    func_interp = interp1d(cdf, xs, bounds_error=False, fill_value=0)
    sample = func_interp(u)
    return sample


def norm_pz(zred_mini, zred_maxi, zreds, pdf_zred):
    """normalize int_{zred_mini}^{zred_maxi} p(z) = 1"""
    idx_zrange = np.logical_and(zreds >= zred_mini, zreds <= zred_maxi)
    zreds_inrange = zreds[idx_zrange]
    p_int = simpson(y=pdf_zred[idx_zrange], x=zreds_inrange)
    pdf_zred_inrange = pdf_zred[idx_zrange] / p_int
    invalid = np.where(pdf_zred_inrange < 0)
    pdf_zred_inrange[invalid] = 0
    finterp_z_pdf = interp1d(
        zreds_inrange, pdf_zred_inrange, bounds_error=False, fill_value=0
    )

    cdf_zred = np.cumsum(pdf_zred_inrange)
    cdf_zred /= max(cdf_zred)
    invalid = np.where(cdf_zred < 0)
    cdf_zred[invalid] = 0
    cdf_zred[-1] = 1
    finterp_cdf_z = interp1d(cdf_zred, zreds_inrange, bounds_error=False, fill_value=0)

    return (finterp_z_pdf, finterp_cdf_z)


############ Needed for SFH(M,z) ############


def z_to_agebins_rescale(zstart, nbins_sfh=7, amin=7.1295):
    """agelims here must match those in z_to_agebins(), which set the nonparameteric
    SFH age bins depending on the age of the universe at a given z.

    This function ensures that the agebins used for calculating the expectation values of logsfr_ratios
    follow the same spacing.
    """

    agelims = np.zeros(nbins_sfh + 1)
    agelims[0] = (
        cosmo.lookback_time(zstart).to(u.yr).value
    )  # shift the start of the agebin
    tuniv = (
        cosmo.lookback_time(15).to(u.yr).value
    )  # cap at z~15, for the onset of star formation
    tbinmax = tuniv - (tuniv - agelims[0]) * 0.10
    agelims[-2] = tbinmax
    agelims[-1] = tuniv

    if zstart <= 3.0:
        agelims[1] = agelims[0] + 3e7  # 1st bin is 30 Myr wide
        agelims[2] = agelims[1] + 1e8  # 2nd bin is 100 Myr wide
        i_age = 3
        nbins = len(agelims) - 3
    else:
        agelims[1] = agelims[0] + 10**amin
        i_age = 2
        nbins = len(agelims) - 2

    if agelims[0] == 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            agelims = (
                np.log10(agelims[:i_age]).tolist()[:-1]
                + np.squeeze(
                    np.linspace(np.log10(agelims[i_age - 1]), np.log10(tbinmax), nbins)
                ).tolist()
                + [np.log10(tuniv)]
            )
            agelims[0] = 0

    else:
        agelims = (
            np.log10(agelims[:i_age]).tolist()[:-1]
            + np.squeeze(
                np.linspace(np.log10(agelims[i_age - 1]), np.log10(tbinmax), nbins)
            ).tolist()
            + [np.log10(tuniv)]
        )

    agebins = np.array([agelims[:-1], agelims[1:]]).T
    return 10**agebins


### functions needed for the SFH(M,z) prior
def slope(x, y):
    return (y[1] - y[0]) / (x[1] - x[0])


def slope_and_intercept(x, y):
    a = slope(x, y)
    b = -x[0] * a + y[0]
    return a, b


def delta_t_dex(m, mlims=[9, 12], dlims=[-0.2, 0.8]):
    """Introduces mass dependence in SFH"""
    a, b = slope_and_intercept([mlims[0], mlims[1]], [dlims[0], dlims[1]])

    if m <= mlims[0]:
        return dlims[0]
    elif m >= mlims[1]:
        return dlims[1]
    else:
        return a * m + b


def expe_logsfr_ratios(
    this_z, this_m, logsfr_ratio_mini, logsfr_ratio_maxi, nbins_sfh=7, amin=7.1295
):
    """expectation values of logsfr_ratios"""

    age_shifted = np.log10(cosmo.age(this_z).value) + delta_t_dex(this_m)
    age_shifted = 10**age_shifted

    zmin_thres = 0.15
    zmax_thres = 10
    if age_shifted < AGE_GRID[-1]:
        z_shifted = zmax_thres * 1
    elif age_shifted > AGE_GRID[0]:
        z_shifted = zmin_thres * 1
    else:
        z_shifted = F_AGE_Z(age_shifted)
    if z_shifted > zmax_thres:
        z_shifted = zmax_thres * 1
    if z_shifted < zmin_thres:
        z_shifted = zmin_thres * 1

    agebins_shifted = z_to_agebins_rescale(
        zstart=z_shifted, nbins_sfh=nbins_sfh, amin=amin
    )

    nsfrbins = agebins_shifted.shape[0]
    sfr_shifted = np.zeros(nsfrbins)
    for i in range(nsfrbins):
        a = agebins_shifted[i, 0]
        b = agebins_shifted[i, 1]
        sfr_shifted[i] = SPL_TL_SFRD.integral(a=a, b=b) / (b - a)

    logsfr_ratios_shifted = np.zeros(nsfrbins - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(nsfrbins - 1):
            logsfr_ratios_shifted[i] = np.log10(sfr_shifted[i] / sfr_shifted[i + 1])
    logsfr_ratios_shifted = np.clip(
        logsfr_ratios_shifted, logsfr_ratio_mini, logsfr_ratio_maxi
    )

    if not np.all(np.isfinite(logsfr_ratios_shifted)):
        # Identify indices where values are NaN
        nan_mask = np.isnan(logsfr_ratios_shifted)

        # Find the first index that is NaN
        bad_indices = np.where(nan_mask)[0]

        if len(bad_indices) > 0:
            first_nan_idx = np.min(bad_indices)

            # Ensure there is a neighbor to the left to copy from
            if first_nan_idx > 0:
                neighbor_val = logsfr_ratios_shifted[first_nan_idx - 1]

                # Forward fill the rest of the array with this neighbor value
                logsfr_ratios_shifted[first_nan_idx:] = neighbor_val

    return logsfr_ratios_shifted
