"""This module contains parameter transformations that may be useful to
transform from parameters that easier to sample in to the parameters required
for building SED models.

They can be used as ``"depends_on"`` entries in parameter specifications.
"""

import numpy as np
from ..sources.constants import cosmo

__all__ = ["stellar_logzsol", "delogify_mass",
           "tburst_from_fage", "tage_from_tuniv", "zred_to_agebins",
           "dustratio_to_dust1",
           "zfrac_to_masses", "zfrac_to_sfrac", "zfrac_to_sfr", "masses_to_zfrac",
           "sfratio_to_sfr", "sfratio_to_mass"]


# --------------------------------------
# --- Basic Convenience Transforms ---
# --------------------------------------

def stellar_logzsol(logzsol=0.0, **extras):
    """Simple function that takes an argument list and returns the value of the
    `logzsol` argument (i.e. the stellar metallicity)

    :param logzsol:
        FSPS stellar metaliicity parameter.

    :returns logzsol:
        The same.
    """
    return logzsol


def delogify_mass(logmass=0.0, **extras):
    """Simple function that takes an argument list including a `logmass`
    parameter and returns the corresponding linear mass.

    :param logmass:
        The log10(mass)

    :returns mass:
        The mass in linear units
    """
    return 10**logmass

def total_mass(mass=0.0, **extras):
    """Simple function that takes an argument list uncluding a `mass`
    parameter and returns the corresponding total mass.

    :param mass:
        length-N vector of masses in bins

    :returns total mass:
        Total mass in linear units
    """
    return mass.sum()

# --------------------------------------
# Fancier transforms
# --------------------------------------

def tburst_from_fage(tage=0.0, fage_burst=0.0, **extras):
    """This function transfroms from a fractional age of a burst to an absolute
    age.  With this transformation one can sample in ``fage_burst`` without
    worry about the case ``tburst`` > ``tage``.

    :param tage:
        The age of the host galaxy (Gyr)

    :param fage_burst:
        The fraction of the host age at which the burst occurred

    :returns tburst:
        The age of the host when the burst occurred (i.e. the FSPS ``tburst``
        parameter)
    """
    return tage * fage_burst


def tage_from_tuniv(zred=0.0, tage_tuniv=1.0, **extras):
    """This function calculates a galaxy age from the age of the univers at
    ``zred`` and the age given as a fraction of the age of the universe.  This
    allows for both ``zred`` and ``tage`` parameters without ``tage`` exceeding
    the age of the universe.

    :param zred:
        Cosmological redshift.

    :param tage_tuniv:
        The ratio of ``tage`` to the age of the universe at ``zred``.

    :returns tage:
        The stellar population age, in Gyr
    """
    tuniv = cosmo.age(zred).value
    tage = tage_tuniv * tuniv
    return tage


def zred_to_agebins(zred=0.0, agebins=[], **extras):
    """Set the nonparameteric SFH age bins depending on the age of the universe
    at ``zred``. The first bin is not altered and the last bin is always 15% of
    the upper edge of the oldest bin, but the intervenening bins are evenly
    spaced in log(age).

    :param zred:
        Cosmological redshift.  This sets the age of the universe.

    :param agebins:
        The SFH bin edges in log10(years).  ndarray of shape ``(nbin, 2)``.

    :returns agebins:
        The new SFH bin edges.
    """
    tuniv = cosmo.age(zred).value * 1e9
    tbinmax = tuniv * 0.85
    ncomp = len(agebins)
    agelims = list(agebins[0]) + np.linspace(agebins[1][1], np.log10(tbinmax), ncomp-2).tolist() + [np.log10(tuniv)]
    return np.array([agelims[:-1], agelims[1:]]).T


def dustratio_to_dust1(dust2=0.0, dust_ratio=0.0, **extras):
    """Set the value of dust1 from the value of dust2 and dust_ratio

    :param dust2:
        The diffuse dust V-band optical depth (the FSPS ``dust2`` parameter.)

    :param dust_ratio:
        The ratio of the extra optical depth towards young stars to the diffuse
        optical depth affecting all stars.

    :returns dust1:
        The extra optical depth towards young stars (the FSPS ``dust1``
        parameter.)
    """
    return dust2 * dust_ratio

# --------------------------------------
# --- Transforms for the continuity non-parametric SFHs used in (Leja et al. 2018) ---
# --------------------------------------

def logsfr_ratios_to_masses(logmass=None, logsfr_ratios=None, agebins=None, **extras):
    logsfr_ratios = np.clip(logsfr_ratios,-100,100) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()

    return m1 * coeffs

def logsfr_ratios_to_masses_flex(logmass=None, logsfr_ratio_young=None, logsfr_ratio_old=None, agebins=None, **extras):
    logsfr_ratio_young = np.clip(logsfr_ratio_young,-100,100)
    logsfr_ratio_old = np.clip(logsfr_ratio_old,-100,100)

    nbins = agebins.shape[0]-2
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    dtyoung, dt1 = (10**agebins[:2,1]-10**agebins[:2,0])
    dtn, dtold = (10**agebins[-2:,1]-10**agebins[-2:,0])
    mbin = (10**logmass) / (syoung*dtyoung/dt1 + sold*dtold/dtn + nbins)
    myoung = syoung*mbin*dtyoung/dt1
    mold = sold*mbin*dtold/dtn
    n_masses = np.full(nbins, mbin)

    return np.array(myoung.tolist()+n_masses.tolist()+mold.tolist())

def logsfr_ratios_to_agebins(logsfr_ratios=None, **extras):
    """this transforms from SFR ratios to agebins
    by assuming a constant amount of mass forms in each bin
    agebins = np.array([NBINS,2])

    use equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin
    """

    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios,-100,100)

    # calculate delta(t) for oldest, youngest bins (fixed)
    lower_time = (10**agebins[0,1]-10**agebins[0,0])
    upper_time = (10**agebins[-1,1]-10**agebins[-1,0])
    tflex = (10**agebins[-1,-1]-upper_time-lower_time)

    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tflex / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))

    # translate into agelims vector (time bin edges)
    agelims = [1, lower_time, dt1+lower_time]
    for i in range(n_ratio): agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    agelims += [tuniv[0]]
    agebins = np.log10([agelims[:-1], agelims[1:]]).T

    return agebins

# --------------------------------------
# --- Transforms for Dirichlet non-parametric SFH used in (Leja et al. 2017) ---
# --------------------------------------

def zfrac_to_sfrac(z_fraction=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions. The transformation is such that sfr fractions are drawn from a
    Dirichlet prior.  See Betancourt et al. 2010 and Leja et al. 2017

    :param z_fraction:
        latent variables drawn form a specific set of Beta distributions. (see
        Betancourt 2010)

    :returns sfrac:
        The star formation fractions (See Leja et al. 2017 for definition).
    """
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    return sfr_fraction


def zfrac_to_masses(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions and then to bin mass fractions. The transformation is such that
    sfr fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010
    and Leja et al. 2017

    :param total_mass:
        The total mass formed over all bins in the SFH.

    :param z_fraction:
        latent variables drawn form a specific set of Beta distributions. (see
        Betancourt 2010)

    :returns masses:
        The stellar mass formed in each age bin.
    """
    # sfr fractions
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    masses = total_mass * mass_fraction
    return masses


def zfrac_to_sfr(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to SFRs.

    :returns sfrs:
        The SFR in each age bin (msun/yr).
    """
    # sfr fractions
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    masses = total_mass * mass_fraction
    return masses / time_per_bin


def masses_to_zfrac(mass=None, agebins=None, **extras):
    """The inverse of :py:meth:`zfrac_to_masses`, for setting mock parameters
    based on mock bin masses.

    :returns total_mass:
        The total mass

    :returns zfrac:
        The dimensionless `z` variables used for sfr fraction parameterization.
    """
    total_mass = mass.sum()
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    sfr_fraction = mass / time_per_bin
    sfr_fraction /= sfr_fraction.sum()

    z_fraction = np.zeros(len(sfr_fraction) - 1)
    z_fraction[0] = 1 - sfr_fraction[0]
    for i in range(1, len(z_fraction)):
        z_fraction[i] = 1.0 - sfr_fraction[i] / np.prod(z_fraction[:i])

    return total_mass, z_fraction


# --------------------------------------
# --- Transforms for SFR ratio based nonparameteric SFH ---
# --------------------------------------

def sfratio_to_sfr(sfr_ratio=None, sfr0=None, **extras):
    sfr = np.cumprod(sfr_ratio)
    all_sfr = np.insert(sfr*sfr0, sfr0, 0)
    return all_sfr


def sfratio_to_mass(sfr_ratio=None, sfr0=None, agebins=None, **extras):
    sfr = sfratio_to_sfr(sfr_ratio=sfr_ratio, sfr0=sfr0)
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass = sfr * time_per_bin
    return mass
