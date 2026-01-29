#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
priors_beta.py

This module implements the Prospector-:math:`\\beta` priors, as described in `Wang et al. (2023)`_.
These priors are designed to incorporate physically motivated constraints on galaxy
properties, specifically:

1.  **Stellar Mass Function (SMF)**: Constrains the galaxy stellar mass distribution
    based on the continuity model from `Leja et al. (2020)`_ at :math:`z < 3` and
    optionally `Tacchella et al. (2018)`_ at higher redshifts (``const_phi = False``).
2.  **Mass-Metallicity Relation (MZR)**: Constrains the stellar metallicity
    as a function of stellar mass, based on `Gallazzi et al. (2005)`_.
3.  **Dynamic Redshift Prior**: Combines the galaxy number density (derived from
    the SMF) with the differential comoving volume to produce a physically
    meaningful redshift prior, :math:`p(z) \\sim N(z) \\frac{dV}{dz}`.
4.  **Star Formation History (SFH) Prior**: Constrains the shape of the SFH
    using the cosmic star formation rate density (CSFRD) from `Behroozi et al. (2019)`_.

The module provides several classes that combine these components in different ways:

* ``PhiMet``: SMF + MZR. Uniform redshift prior.
* ``ZredMassMet``: Dynamic :math:`p(z)` + SMF + MZR.
* ``DymSFH``: MZR + SFH prior. Uniform mass and redshift priors.
* ``DymSFHfixZred``: Same as ``DymSFH`` but with fixed redshift.
* ``PhiSFH``: SMF + MZR + SFH prior. Uniform redshift prior.
* ``PhiSFHfixZred``: Same as ``PhiSFH`` but with fixed redshift.
* ``NzSFH``: The full Prospector-:math:`\\beta` prior set: Dynamic :math:`p(z)` + SMF + MZR + SFH prior.

All classes inherit from ``BetaPrior``, which provides the shared infrastructure
for initialization, probability calculation (``__call__``), sampling (``sample``),
and unit transformation (``unit_transform``).

.. _Wang et al. (2023): https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..58W
.. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
.. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
.. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
.. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict

try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
from scipy.interpolate import interp1d
from scipy.stats import t
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u

from . import priors
from .prior_data import (
    MASSMET,
    SPL_TL_SFRD,
    F_AGE_Z,
    WMAP9_AGE,
)

__all__ = [
    "BetaPrior",
    "PhiMet",
    "ZredMassMet",
    "DymSFH",
    "DymSFHfixZred",
    "PhiSFH",
    "PhiSFHfixZred",
    "NzSFH",
]

_, AGE_GRID = WMAP9_AGE

# Continuity model median parameters + 1-sigma uncertainties (Leja+20).
LEJA_20_PARS = {
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


class BetaPrior(priors.Prior):
    """
    Base class for Prospector-:math:`\\beta` priors.

    This class implements the core logic for the physically motivated priors,
    handling the initialization of distributions, calculation of probabilities,
    and sampling. It supports different combinations of redshift, mass,
    metallicity, and SFH priors.

    The actual prior classes (e.g., ``NzSFH``) subclass this and configure the
    flags (``z_prior``, ``mass_prior``, ``sfh_prior``) appropriately.

    Methods
    -------
    __init__(parnames, name, z_prior, mass_prior, sfh_prior, **kwargs)
        Initialize the BetaPrior.
    __call__(x, **kwargs)
        Compute the natural log of the prior probability.
    __len__()
        Return the number of parameters.
    sample(nsample=None, **kwargs)
        Draw samples from the prior distribution.
    unit_transform(x, **kwargs)
        Transform from the unit hypercube to the physical parameter space.
    """

    # Hardcoded grid sizes for various integrations and interpolations
    ZRED_GRID_LEN = 4_000
    MASS_GRID_LEN = 101
    MASS_INTEG_GRID = 100
    MASS_SUBGRID_LEN = 1_000

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
        parnames: List[str] = [],
        name: str = "",
        z_prior: str = "uniform",  # 'uniform', 'fixed', 'dynamic'
        mass_prior: str = "uniform",  # 'uniform', 'mass_function'
        sfh_prior: bool = False,  # True/False
        **kwargs,
    ):
        """
        Initialize the BetaPrior.

        Parameters
        ----------
        parnames : List[str]
            List of parameter names to be used for aliasing. If empty, defaults
            to ``self.prior_params``.
        name : str
            Name of the prior.
        z_prior : str, optional
            Type of redshift prior:
            - ``'uniform'``: Uniform distribution between ``zred_mini`` and ``zred_maxi``.
            - ``'fixed'``: Redshift is fixed to ``zred``.
            - ``'dynamic'``: Calculated from number density and comoving volume.
        mass_prior : str, optional
            Type of mass prior:
            - ``'uniform'``: Uniform distribution between ``mass_mini`` and ``mass_maxi`` (in log mass).
            - ``'mass_function'``: Based on the stellar mass function.
        sfh_prior : bool, optional
            Whether to include the SFH prior based on cosmic SFRD.
        **kwargs
            Additional keyword arguments to set prior parameters (e.g., ``zred_mini=0.0``).
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
                    self.params.get("mass_mini", 9.0),
                    self.params["mass_maxi"],
                    self.MASS_GRID_LEN,
                )
            else:
                self.mgrid = None
        else:
            raise ValueError(f"Unknown mass_prior type: {self.mass_prior_type}")

        # --- SFH Prior Setup ---
        if self.sfh_prior_flag:
            self.logsfr_ratios_dist = priors.FastTruncatedEvenStudentTFreeDeg2(
                hw=self.params["logsfr_ratio_maxi"],
                sig=self.params["logsfr_ratio_tscale"],
            )

    def _setup_dynamic_z_prior_from_norm(self):
        """
        Constructs :math:`p(z) \\sim N(z) \\frac{dV}{dz}` using pre-calculated :math:`N(z)`.

        :math:`N(z)` is the integral of the mass function above the mass limit at redshift :math:`z`.
        :math:`\\frac{dV}{dz}` is the differential comoving volume.
        The resulting PDF is normalized and inverted to create a CDF for sampling.
        """

        dvol = cosmo.differential_comoving_volume(self._z_grid_norm).value
        pdf_z_unnorm = self._n_gal_z * dvol

        self.finterp_z_pdf, self.finterp_cdf_z = norm_pz(
            self.params["zred_mini"],
            self.params["zred_maxi"],
            self._z_grid_norm,
            pdf_z_unnorm,
        )

    def _setup_mass_normalization(self):
        """
        Calculates the integral of the mass function over the mass limits
        as a function of redshift.

        This computes :math:`N(z) = \\int_{M_\\textrm{min}(z)}^{M_\\textrm{max}} \\Phi(M, z)\\,dM`.
        This quantity is used for:
        1. Normalizing the mass prior :math:`p(M \\vert z) = \\Phi(M, z) / N(z)`.
        2. Constructing the dynamic redshift prior :math:`p(z) \\sim N(z) \\frac{dV}{dz}`.

        The result is stored as ``self.finterp_mass_norm``, an interpolator over :math:`z`.
        """

        # Define grid (high res for accuracy)
        self._z_grid_norm = np.linspace(
            self.params.get("zred_mini", 0),
            self.params.get("zred_maxi", 10),
            self.ZRED_GRID_LEN,
        )

        n_gal_z = []
        mass_max = self.params["mass_maxi"]

        for z in self._z_grid_norm:
            m_min = self.mass_min_func(z)
            if m_min >= mass_max:
                n_gal_z.append(0.0)
                continue

            m_integ_grid = np.linspace(m_min, mass_max, self.MASS_INTEG_GRID)
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

    def __len__(self) -> int:
        """
        Returns the number of parameters managed by this prior.

        If SFH prior is enabled, it includes redshift, mass, metallicity, and ``nbins_sfh - 1``
        SFH parameters. Otherwise, it returns 3 (redshift, mass, metallicity). If redshift is fixed,
        the redshift parameter is not counted.
        """
        n_params = 2  # mass and metallicity always included

        if self.z_prior_type != "fixed":
            n_params += 1

        if self.sfh_prior_flag:
            n_params += self.params["nbins_sfh"] - 1

        return n_params

    @property
    def range(self) -> Tuple[Tuple[float, float], ...]:
        """
        Returns the valid range for each parameter.
        """
        r = []
        # Z range
        if self.z_prior_type == "fixed":
            pass
        else:
            r.append((self.params["zred_mini"], self.params["zred_maxi"]))

        r.append((self.params.get("mass_mini", 9.0), self.params["mass_maxi"]))
        r.append((self.params["z_mini"], self.params["z_maxi"]))

        if self.sfh_prior_flag:
            # SFH ratios have the same range
            r.append(
                (self.params["logsfr_ratio_mini"], self.params["logsfr_ratio_maxi"])
            )

        return tuple(r)

    def bounds(self, **kwargs) -> Tuple[Tuple[float, float], ...]:
        """
        Returns the bounds of the parameters.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Compute the natural log of the prior probability.

        Parameters
        ----------
        x : np.ndarray
            Input parameter vector(s). Shape ``(N_params,)`` or ``(N_samples, N_params)``.
            The order is expected to be:
            0: Redshift
            1: Log Stellar Mass
            2: Log Stellar Metallicity (Solar units)
            3...: Log SFR ratios (if SFH prior is enabled)

        Returns
        -------
        lnp : float or np.ndarray
            The natural log probability density.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        if x.ndim == 1:
            x = x[np.newaxis, :]
            scalar_input = True
        else:
            scalar_input = False

        # Determine indices based on whether redshift is fixed
        if self.z_prior_type == "fixed":
            idx_z = None
            idx_m = 0
            idx_met = 1
            idx_sfh = 2
        else:
            idx_z = 0
            idx_m = 1
            idx_met = 2
            idx_sfh = 3

        if idx_z is not None:
            zreds = x[..., idx_z]
        else:
            # Broadcast self.zred to shape of x excluding last dim
            shape = x.shape[:-1]
            zreds = np.full(shape, self.zred)

        logms = x[..., idx_m]
        logzsols = x[..., idx_met]

        lnp = np.zeros_like(x)  # Shape (N, N_params)

        # --- 1. Redshift Prior ---
        if idx_z is not None:
            if self.z_prior_type == "uniform":
                lnp[..., idx_z] = self.zred_dist(zreds)
            elif self.z_prior_type == "dynamic":
                # vectorizing finterp_z_pdf
                # finterp_z_pdf is from interp1d, which should handle vectors
                val = self.finterp_z_pdf(zreds)
                val = np.maximum(val, 1e-300)  # Avoid log(0)
                lnp[..., idx_z] = np.log(val)

        # --- 2. Mass Prior ---
        if self.mass_prior_type == "uniform":
            lnp[..., idx_m] = self.mass_dist(logms)
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
                lnp[..., idx_m] = np.log(p_mass)

        # --- 3. Metallicity Prior ---
        # p(zsol) depends on mass
        # We calculate this per sample because parameters depend on mass.

        # Ensure intputs are iterable
        logms_iterable = np.atleast_1d(logms)
        logzsols_iterable = np.atleast_1d(logzsols)

        # Calculate parameters
        mus = loc_massmet(logms_iterable)
        sigs = scale_massmet(logms_iterable)

        # Create distributions
        met_dists = [
            priors.FastTruncatedNormal(
                a=self.params["z_mini"],
                b=self.params["z_maxi"],
                mu=m,
                sig=s,
            )
            for m, s in zip(mus, sigs)
        ]

        # Calculate probabilities using the iterable logzsols
        lnp[..., idx_met] = [
            dist(zsol) for dist, zsol in zip(met_dists, logzsols_iterable)
        ]

        # --- 4. SFH Prior ---
        if self.sfh_prior_flag:
            logsfr_input = x[..., idx_sfh:]
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
                lnp[..., idx_sfh:] = np.log(p_sfh)

        if scalar_input:
            return lnp[0]
        else:
            return lnp

    def sample(self, nsample: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Draw samples from the prior distribution.

        Parameters
        ----------
        nsample : int, optional
            Number of samples to draw. If ``None``, draws 1 sample.
        **kwargs
            Update prior parameters before sampling.

        Returns
        -------
        samples : np.ndarray
            Array of samples. Shape ``(N_params,)`` if ``nsample`` is ``None``,
            else ``(N_samples, N_params)``.
        """
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
                m_subgrid = np.linspace(m_min, m_max, self.MASS_SUBGRID_LEN)

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
            res = []
            if self.z_prior_type != "fixed":
                res.append(np.atleast_1d(zred))
            res.append(np.atleast_1d(mass[0]))
            res.append(np.atleast_1d(met[0]))
            if self.sfh_prior_flag:
                res.append(np.atleast_1d(sfh_rvs[0]))
            return np.concatenate(res)
        else:
            # vstack
            res = []
            if self.z_prior_type != "fixed":
                res.append(zred)
            res.append(mass)
            res.append(met)
            if self.sfh_prior_flag:
                # sfh_rvs is (n, bins) -> transpose to (bins, n) for vstack
                res.append(sfh_rvs.T)
            return np.vstack(res).T

    def unit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform from the unit hypercube to the physical parameter space.

        Used for nested sampling (e.g. dynesty).

        Parameters
        ----------
        x : np.ndarray
            Input array of uniform random variables in :math:`[0, 1]`. Shape ``(N_params,)``.
        **kwargs
            Update parameters.

        Returns
        -------
        params : np.ndarray
            The physical parameters corresponding to ``x``.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        ptr = 0

        # x is 1D array of length N_params (unit cube coordinates)

        # 1. Redshift
        zred_val = 0.0
        if self.z_prior_type == "fixed":
            zred_val = self.zred * 1
        else:
            if self.z_prior_type == "uniform":
                zred_val = self.zred_dist.unit_transform(x[ptr])
            elif self.z_prior_type == "dynamic":
                zred_val = self.finterp_cdf_z(x[ptr])
            ptr += 1

        # 2. Mass
        mass_val = 0.0
        if self.mass_prior_type == "uniform":
            mass_val = self.mass_dist.unit_transform(x[ptr])
        elif self.mass_prior_type == "mass_function":
            m_min = self.mass_min_func(zred_val)
            m_max = self.params["mass_maxi"]
            m_subgrid = np.linspace(m_min, m_max, self.MASS_SUBGRID_LEN)

            cdf_mass = cdf_mass_func_at_z(
                z=zred_val,
                logm=m_subgrid,
                const_phi=self.params["const_phi"],
                bounds=[m_min, m_max],
            )
            mass_val = ppf(x[ptr], m_subgrid, cdf=cdf_mass)
        ptr += 1

        # 3. Metallicity
        met_dist = priors.FastTruncatedNormal(
            a=self.params["z_mini"],
            b=self.params["z_maxi"],
            mu=loc_massmet(mass_val),
            sig=scale_massmet(mass_val),
        )
        met_val = met_dist.unit_transform(x[ptr])
        ptr += 1

        res = []
        if self.z_prior_type != "fixed":
            res.append(np.atleast_1d(zred_val))
        res.append(np.atleast_1d(mass_val))
        res.append(np.atleast_1d(met_val))

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
            # x[ptr:] corresponds to sfh bins
            sfh_unit = x[ptr:]
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
    """
    BetaPrior configuration: SMF + MZR.

    - **Redshift**: Uniform
    - **Mass**: Stellar Mass Function (`Leja et al. (2020)`_ / `Tacchella et al. (2018)`_)
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
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
    """
    BetaPrior configuration: Dynamic :math:`p(z)` + SMF + MZR.

    - **Redshift**: Dynamic (Number Density * dV/dz)
    - **Mass**: Stellar Mass Function (`Leja et al. (2020)`_ / `Tacchella et al. (2018)`_)
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
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
    """
    BetaPrior configuration: MZR + SFH Prior.

    - **Redshift**: Uniform
    - **Mass**: Uniform
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)
    - **SFH**: Dynamic SFH prior based on cosmic SFRD (`Behroozi et al. (2019)`_)

    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    .. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
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
    """
    BetaPrior configuration: MZR + SFH Prior (Fixed Redshift).

    - **Redshift**: Fixed
    - **Mass**: Uniform
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)
    - **SFH**: Dynamic SFH prior based on cosmic SFRD (`Behroozi et al. (2019)`_)

    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    .. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
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
    """
    BetaPrior configuration: SMF + MZR + SFH Prior.

    - **Redshift**: Uniform
    - **Mass**: Stellar Mass Function (`Leja et al. (2020)`_ / `Tacchella et al. (2018)`_)
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)
    - **SFH**: Dynamic SFH prior based on cosmic SFRD (`Behroozi et al. (2019)`_)

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    .. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
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
    """
    BetaPrior configuration: SMF + MZR + SFH Prior (Fixed Redshift).

    - **Redshift**: Fixed
    - **Mass**: Stellar Mass Function (`Leja et al. (2020)`_ / `Tacchella et al. (2018)`_)
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)
    - **SFH**: Dynamic SFH prior based on cosmic SFRD (`Behroozi et al. (2019)`_)

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    .. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
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
    """
    BetaPrior configuration: Full Prospector-:math:`\\beta` (Dynamic :math:`p(z)` + SMF + MZR + SFH).

    - **Redshift**: Dynamic (Number Density * dV/dz)
    - **Mass**: Stellar Mass Function (`Leja et al. (2020)`_ / `Tacchella et al. (2018)`_)
    - **Metallicity**: Mass-Metallicity Relation (`Gallazzi et al. (2005)`_)
    - **SFH**: Dynamic SFH prior based on cosmic SFRD (`Behroozi et al. (2019)`_)

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    .. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
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


def scale_massmet(mass: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the standard deviation of the Gaussian approximating the
    mass-metallicity relationship at a given mass.

    Interpolates between the 16th and 84th percentiles of the `Gallazzi et al. (2005)`_ relation.

    Parameters
    ----------
    mass : float or array_like
        Log stellar mass.

    Returns
    -------
    scale : float or array_like
        The width (sigma) of the metallicity distribution.

    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    """
    upper_84 = np.interp(mass, MASSMET[:, 0], MASSMET[:, 3])
    lower_16 = np.interp(mass, MASSMET[:, 0], MASSMET[:, 2])
    return upper_84 - lower_16


def loc_massmet(mass: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the mean of the Gaussian approximating the mass-metallicity
    relationship at a given mass.

    Interpolates the mean of the `Gallazzi et al. (2005)`_ relation.

    Parameters
    ----------
    mass : float or array_like
        Log stellar mass.

    Returns
    -------
    loc : float or array_like
        The mean log metallicity.

    .. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
    """
    return np.interp(mass, MASSMET[:, 0], MASSMET[:, 1])


############ Mass function in Leja+20    ############
############ Code modified from appendix ############
def schechter(
    logm: Union[float, np.ndarray],
    logphi: float,
    logmstar: float,
    alpha: float,
    m_lower: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Generate a `Schechter (1976)`_ function (in dlogm).

    .. math::
        \\Phi(M) = \\ln(10) \\Phi^* 10^{(M-M^*)(\\alpha+1)} \\exp(-10^{M-M^*})

    Parameters
    ----------
    logm : float or array_like
        Log stellar mass.
    logphi : float
        Log normalization (:math:`\\Phi^*`).
    logmstar : float
        Log characteristic mass (:math:`M^*`).
    alpha : float
        Low-mass slope.
    m_lower : float, optional
        Unused, but kept for signature compatibility.

    Returns
    -------
    phi : float or array_like
        The value of the mass function.

    .. _Schechter (1976): https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S
    """
    phi = (
        (10**logphi)
        * np.log(10)
        * 10 ** ((logm - logmstar) * (alpha + 1))
        * np.exp(-(10 ** (logm - logmstar)))
    )
    return phi


def parameter_at_z0(
    y: Tuple[float, float, float],
    z0: float,
    z1: float = 0.2,
    z2: float = 1.6,
    z3: float = 3.0,
) -> float:
    """
    Compute parameter at redshift ``z0`` using a quadratic interpolation
    between three anchor points (``z1``, ``z2``, ``z3``).

    Used for evolving the `Schechter (1976)`_ function parameters in
    the `Leja et al. (2020)`_ continuity model.

    Parameters
    ----------
    y : tuple
        The parameter values at ``z1``, ``z2``, and ``z3``.
    z0 : float
        The target redshift.
    z1, z2, z3 : float
        The anchor redshifts.

    Returns
    -------
    val : float
        The interpolated parameter value at ``z0``.

    .. _Schechter (1976): https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S
    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    """
    y1, y2, y3 = y
    a = ((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (
        z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)
    )
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c


def draw_at_z(
    z0: Union[float, np.ndarray] = 1.0,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Get the mass function parameters at redshift ``z0``.

    The `Leja et al. (2020)`_ mass function is defined over :math:`0.2 \\leq z \\leq 3.0`.
    Outside this range, the values are clamped to the boundaries.

    Parameters
    ----------
    z0 : float or array_like
        Redshift.

    Returns
    -------
    draws : dict
        Dictionary containing ``'logphi1'``, ``'logphi2'``, ``'logmstar'``, ``'alpha1'``, ``'alpha2'``.

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
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
        samp = LEJA_20_PARS[par]
        if par in ["logphi1", "logphi2", "logmstar"]:
            draws[par] = parameter_at_z0(samp, _z0)
        else:
            draws[par] = np.array(samp)

    return draws


def low_z_mass_func(
    z0: float, logm: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Evaluate the `Leja et al. (2020)`_ double `Schechter (1976)`_ mass function
    at redshift ``z0`` (:math:`z < 3`).

    Parameters
    ----------
    z0 : float
        Redshift.
    logm : float or array_like
        Log stellar mass.

    Returns
    -------
    phi : float or array_like
        The mass function value.

    .. _Schechter (1976): https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S
    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
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
    """:math:`\\cos^2` weighted average

    zrange: min and max of the redshift range over which we will take the :math:`\\cos^2` weighted average
    """

    z_scaled = scale(z, zrange=zrange)
    w_l20 = np.cos(z_scaled) ** 2
    w_t18 = 1 - np.cos(z_scaled) ** 2

    return np.array([w_l20, w_t18])


# Best-fit parameters from Table 2 of Tacchella+18.
z_t18 = np.arange(4, 13, 1)
phi_t18 = np.array([261.9, 201.2, 140.5, 78.0, 38.4, 37.3, 8.1, 3.9, 1.1])
phi_t18 *= 1e-5
logm_t18 = np.array([10.16, 9.89, 9.62, 9.38, 9.18, 8.74, 8.79, 8.50, 8.50])
m_t18 = 10**logm_t18
alpha_t18 = np.array([-1.54, -1.59, -1.64, -1.70, -1.76, -1.80, -1.92, -2.00, -2.10])


def schechter_t18(m, phi_star, m_star, alpha):
    """
    `Schechter (1976)`_ function in linear space (:math:`\\frac{dN}{dM}`), as used in `Tacchella et al. (2018)`_.

    .. _Schechter (1976): https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    """
    return phi_star * (m / m_star) ** (alpha + 1) * np.exp(-m / m_star)


def high_z_mass_func_discreate(z0, this_m):
    """
    Evaluate the `Tacchella et al. (2018)`_ mass function at one of the discrete redshift bins.

    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
    """
    # no boundary check is done here
    idx = np.squeeze(np.where(z_t18 == int(np.round(z0))))
    return schechter_t18(
        m=this_m, phi_star=phi_t18[idx], m_star=m_t18[idx], alpha=alpha_t18[idx]
    )


def high_z_mass_func(z0, this_m):
    """
    Evaluate the high-redshift mass function, interpolating between discrete bins.

    For `4 <= z <= 12`, it interpolates between the integer redshift bins using
    a :math:`\\cos^2` weighted average.
    """
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


def mass_func_at_z(
    z: float,
    this_logm: Union[float, np.ndarray],
    const_phi: bool = False,
    bounds: List[float] = [6.0, 12.5],
) -> Union[float, np.ndarray]:
    """
    Evaluate the composite stellar mass function at redshift :math:`z` and mass `\\log_{10}(M/M_\\odot)`.

    Parameters
    ----------
    z : float
        Redshift.
    this_logm : float or array_like
        Log stellar mass.
    const_phi : bool, optional
        If ``True``, disables redshift evolution outside :math:`0.2 \\leq z \\leq 3.0` (i.e., clamps to boundaries).
        If ``False``, uses `Leja et al. (2020)`_ for :math:`z \\leq 3`, `Tacchella et al. (2018)`_ for :math:`z \\geq 4`, and smooth interpolation in between.
    bounds : list, optional
        [min_logm, max_logm]. Values outside this range return 0.

    Returns
    -------
    phi : float or array_like
        The mass function value.

    .. _Leja et al. (2020): https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L
    .. _Tacchella et al. (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T
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
    """
    Normalized PDF of the mass function at redshift :math:`z`.
    """
    phi_50 = mass_func_at_z(z, logm, const_phi, bounds)
    p_phi_int = simpson(y=phi_50, x=logm)
    pdf_at_m = phi_50 / p_phi_int
    return pdf_at_m


def cdf_mass_func_at_z(z, logm, const_phi, bounds):
    """
    CDF of the mass function at redshift :math:`z`.

    Parameters
    ----------
    logm : array_like
        Grid of log mass values. Must be sorted.
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
    """
    Percent Point Function (Inverse CDF).

    Go from a value ``x`` of the CDF (between 0 and 1) to the corresponding parameter value.
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
    """
    Normalize the redshift PDF and create interpolators for PDF and CDF.

    Parameters
    ----------
    zred_mini, zred_maxi : float
        Redshift bounds.
    zreds : array_like
        Redshift grid.
    pdf_zred : array_like
        Unnormalized PDF values on the grid.

    Returns
    -------
    finterp_z_pdf : interp1d
        Interpolator for :math:`p(z)`.
    finterp_cdf_z : interp1d
        Interpolator for Inverse CDF (for sampling).
    """
    idx_zrange = np.logical_and(zreds >= zred_mini, zreds <= zred_maxi)
    zreds_inrange = zreds[idx_zrange]

    # Error if zred range is outside the range of zreds
    if len(zreds_inrange) == 0:
        raise ValueError(
            f"Redshift range [{zred_mini}, {zred_maxi}] is outside the range of zreds."
        )

    p_int = simpson(y=pdf_zred[idx_zrange], x=zreds_inrange)
    pdf_zred_inrange = pdf_zred[idx_zrange] / p_int
    invalid = np.where(pdf_zred_inrange < 0)
    pdf_zred_inrange[invalid] = 0
    finterp_z_pdf = interp1d(
        zreds_inrange, pdf_zred_inrange, bounds_error=False, fill_value=0
    )

    cdf_zred = np.cumsum(pdf_zred_inrange)
    cdf_zred /= cdf_zred[-1]

    # Prepend 0 to CDF and start z to align for interpolation
    cdf_zred = np.concatenate(([0], cdf_zred))
    zreds_cdf = np.concatenate(([zreds_inrange[0]], zreds_inrange))

    # Clean up bounds
    cdf_zred[cdf_zred < 0] = 0
    cdf_zred[-1] = 1

    # Truncate tails to avoid sampling from zero-probability regions
    # 1. Find the effective start (where CDF > epsilon)
    #    We back up one step to include the 0.0 point just before probability starts.
    first_pos = np.argmax(cdf_zred > 1e-10)
    start_idx = max(0, first_pos - 1)

    # 2. Find the effective end (where CDF reaches ~1)
    #    We keep the first point that is ~1.0.
    last_one = np.argmax(cdf_zred > 1.0 - 1e-10)
    end_idx = last_one + 1 if last_one < len(cdf_zred) else len(cdf_zred)

    cdf_zred_trunc = cdf_zred[start_idx:end_idx]
    zreds_cdf_trunc = zreds_cdf[start_idx:end_idx]

    # Ensure endpoints are strictly 0 and 1
    if len(cdf_zred_trunc) > 0:
        cdf_zred_trunc[0] = 0.0
        cdf_zred_trunc[-1] = 1.0

    # Ensure strictly increasing CDF for interp1d
    # If there are flat spots remaining (e.g. gaps in p(z)), np.unique with return_index
    # keeps the first occurrence, effectively skipping the gap in the inverse map.
    unique_cdf, unique_indices = np.unique(cdf_zred_trunc, return_index=True)
    unique_z = zreds_cdf_trunc[unique_indices]

    finterp_cdf_z = interp1d(
        unique_cdf, unique_z, bounds_error=False, fill_value=(unique_z[0], unique_z[-1])
    )

    return (finterp_z_pdf, finterp_cdf_z)


############ Needed for SFH(M,z) ############


def z_to_agebins_rescale(zstart, nbins_sfh=7, amin=7.1295):
    """
    Calculate age bins for the SFH prior.

    agelims here must match those in ``z_to_agebins()``, which set the nonparameteric
    SFH age bins depending on the age of the universe at a given :math:`z`.

    This function ensures that the agebins used for calculating the expectation values of ``logsfr_ratios``
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
    """
    Introduces mass dependence in SFH by shifting the lookback time.
    """
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
    """
    Calculate the expected log SFR ratios given redshift and mass.

    This uses the cosmic SFRD from `Behroozi et al. (2019)`_, shifted in time
    based on the galaxy's mass (downsizing), to predict the average SFH shape.

    .. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
    """

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
