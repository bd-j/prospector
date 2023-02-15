#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""priors_beta.py -- This module contains the prospector-beta priors.
Ref: Wang, Leja, et al., 2023, ApJL.

Specifically, this module includes the following priors --
1. PhiMet      : p(logM|z)p(Z*|logM), i.e., mass funtion + mass-met
2. ZredMassMet : p(z)p(logM|z)p(Z*|logM), i.e., number density + mass funtion + mass-met
3. PhiSFH      : p(logM|z)p(Z*|logM) & SFH(M, z), i.e., mass funtion + mass-met + SFH
4. NzSFH       : p(z)p(logM|z)p(Z*|logM) & SFH(M, z),
                 i.e., number density + mass funtion + mass-met + SFH;
                 this is the full set of prospector-beta priors.

When called these return the ln-prior-probability, and they can also be used to
construct prior transforms (for nested sampling) and can be sampled from.
"""
import os
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from astropy.cosmology import WMAP9 as cosmos
import astropy.units as u
from scipy.stats import t
from . import priors

__all__ = ["PhiMet", "ZredMassMet", "PhiSFH", "NzSFH"]

prior_data_dir = os.path.join(os.path.dirname(__file__), 'prior_data')
massmet = np.loadtxt( os.path.join(prior_data_dir, 'gallazzi_05_massmet.txt'))
z_b19, tl_b19, sfrd_b19 = np.loadtxt( os.path.join(prior_data_dir, 'behroozi_19_sfrd.txt'), unpack=True)
z_age, age = np.loadtxt( os.path.join(prior_data_dir, 'wmap9_z_age.txt'), unpack=True)

spl_tl_sfrd = UnivariateSpline(tl_b19, sfrd_b19, s=0, ext=1) # lookback time in yrs
f_age_z = interp1d(age, z_age) # age of the universe in Gyr

file_pdf_of_z_l20 = os.path.join(prior_data_dir, 'pdf_of_z_l20.txt')
file_pdf_of_z_l20t18 = os.path.join(prior_data_dir, 'pdf_of_z_l20t18.txt')

############################# p(logM|z)p(Z*|logM) #############################
# mass function & Gaussian metallicity priors

class PhiMet(priors.Prior):

    prior_params = ['zred_mini', 'zred_maxi', 'mass_mini', 'mass_maxi', 'z_mini', 'z_maxi', 'const_phi'] # mass is in log10

    def __init__(self, parnames=[], name='', **kwargs):
        """Overwrites __init__ in the base code Prior

        Parameters
        ----------
        parnames : sequence of strings
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

        self.zred_dist = priors.FastUniform(a=self.params['zred_mini'], b=self.params['zred_maxi'])

        self.mgrid = np.linspace(self.params['mass_mini'], self.params['mass_maxi'], 101)

    def __len__(self):
        """Hack to work with Prospector 0.3
        """
        return 3

    @property
    def range(self):
        return ((self.params['zred_mini'], self.params['zred_maxi']),\
                (self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi'])
               )

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = zred, x[1] = logmass, x[2] = logzsol. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        if x.ndim == 1:
            # doing mcmc; x is [zred, logmass, logzsol]
            p = np.zeros_like(x)
            # p(m)
            p[1] = mass_func_at_z(x[0], x[1], self.params['const_phi'])
            # p(zsol)
            met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                  mu=loc_massmet(x[1]), sig=scale_massmet(x[1]))

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(p)

            # p(z)
            lnp[0] = self.zred_dist(x[0])
            lnp[2] = met_dist(x[2]) # FastTruncatedNormal returns ln(p)
            return lnp

        else:
            # write_hdf5. last step.
            # in prior_product, x is of size (nsamples, npriors)
            # Fast* is not vectorized?
            # so just do a loop here
            _zreds = x[...,0]

            all_p = []
            for i in range(len(_zreds)):
                new_x = x[i]
                p = np.zeros_like(new_x)
                p[1] = mass_func_at_z(new_x[0], new_x[1], self.params['const_phi'])
                all_p.append(p)

            all_p = np.array(all_p)

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(all_p)

            met_dists = [priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                    mu=loc_massmet(mass_i), sig=scale_massmet(mass_i)) for mass_i in x[...,1]]
            lnp[...,0] = self.zred_dist(_zreds)
            lnp[...,2] = [met_dists[i](met_i) for (i, met_i) in enumerate(x[...,2])]

            return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.
        Needed for minimizer.

        :param nsample: (optional)
            Unused. Will not work if nsample > 1 in draw_sample()!
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        # draw a zred from pdf(z)
        zred = self.zred_dist.sample()

        # draw from the mass function at the above zred
        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = draw_sample(xs=self.mgrid, cdf=cdf_mass)

        # given mass from above, draw logzsol
        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.sample()

        return np.array([zred, mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        Needed for nested sampling.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        zred = self.zred_dist.unit_transform(x[0])

        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = ppf(x[1], self.mgrid, cdf=cdf_mass)

        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.unit_transform(x[2])

        return np.array([zred, mass, met])


########################### p(z)p(logM|z)p(Z*|logM) ###########################
# galaxy number density & mass function & Gaussian metallicity priors

class ZredMassMet(priors.Prior):

    prior_params = ['zred_mini', 'zred_maxi', 'mass_mini', 'mass_maxi', 'z_mini', 'z_maxi', 'const_phi'] # mass is in log10

    def __init__(self, parnames=[], name='', **kwargs):
        """Overwrites __init__ in the base code Prior

        Parameters
        ----------
        parnames : sequence of strings
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

        if self.params['const_phi']:
            zreds, pdf_zred = np.loadtxt( file_pdf_of_z_l20, unpack=True)
        else:
            zreds, pdf_zred = np.loadtxt( file_pdf_of_z_l20t18, unpack=True)

        self.finterp_z_pdf, self.finterp_cdf_z = norm_pz(self.params['zred_mini'], self.params['zred_maxi'], zreds, pdf_zred)

        self.mgrid = np.linspace(self.params['mass_mini'], self.params['mass_maxi'], 101)

    def __len__(self):
        """Hack to work with Prospector 0.3
        """
        return 3

    @property
    def range(self):
        return ((self.params['zred_mini'], self.params['zred_maxi']),\
                (self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi'])
               )

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = zred, x[1] = logmass. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        if x.ndim == 1:
            # doing mcmc; x is [zred, logmass, logzsol]
            p = np.zeros_like(x)
            # p(z)
            p[0] = self.finterp_z_pdf(x[0])
            # p(m)
            p[1] = mass_func_at_z(x[0], x[1], self.params['const_phi'])
            # p(zsol)
            met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                  mu=loc_massmet(x[1]), sig=scale_massmet(x[1]))

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(p)

            lnp[2] = met_dist(x[2]) # FastTruncatedNormal returns ln(p)
            return lnp

        else:
            # write_hdf5. last step.
            # in prior_product, x is of size (nsamples, npriors)
            # Fast* is not vectorized?
            # so just do a loop here
            _zreds = x[...,0]

            all_p = []
            for i in range(len(_zreds)):
                new_x = x[i]
                p = np.zeros_like(new_x)
                p[0] = self.finterp_z_pdf(new_x[0])
                p[1] = mass_func_at_z(new_x[0], new_x[1], self.params['const_phi'])

                all_p.append(p)

            all_p = np.array(all_p)

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(all_p)

            met_dists = [priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                    mu=loc_massmet(mass_i), sig=scale_massmet(mass_i)) for mass_i in x[...,1]]
            lnp[...,2] = [met_dists[i](met_i) for (i, met_i) in enumerate(x[...,2])]

            return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.
        Needed for minimizer.

        :param nsample: (optional)
            Unused. Will not work if nsample > 1 in draw_sample()!
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        # draw a zred from pdf(z)
        u = np.random.uniform(0, 1, size=nsample)
        zred = self.finterp_cdf_z(u)

        # draw from the mass function at the above zred
        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = draw_sample(xs=self.mgrid, cdf=cdf_mass)

        # given mass from above, draw logzsol
        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.sample()

        return np.array([zred, mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        Needed for nested sampling.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        zred = self.finterp_cdf_z(x[0])

        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = ppf(x[1], self.mgrid, cdf=cdf_mass)

        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.unit_transform(x[2])


        return np.array([zred, mass, met])


####################### p(logM|z)p(Z*|logM) & SFH(M, z) #######################
# mass function & Gaussian metallicity & SFH(M, z) priors
# expectation value of the nonparametric SFH ~ Behroozi+19 cosmic SFRD

class PhiSFH(priors.Prior):

    prior_params = ['zred_mini', 'zred_maxi', 'mass_mini', 'mass_maxi',
                    'z_mini', 'z_maxi', 'logsfr_ratio_mini', 'logsfr_ratio_maxi',
                    'const_phi'] # mass is in log10

    def __init__(self, parnames=[], name='', **kwargs):
        """Overwrites __init__ in the base code Prior

        Parameters
        ----------
        parnames : sequence of strings
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

        if self.params['const_phi']:
            zreds, pdf_zred = np.loadtxt( file_pdf_of_z_l20, unpack=True)
        else:
            zreds, pdf_zred = np.loadtxt( file_pdf_of_z_l20t18, unpack=True)

        self.finterp_z_pdf, self.finterp_cdf_z = norm_pz(self.params['zred_mini'], self.params['zred_maxi'], zreds, pdf_zred)

        self.mgrid = np.linspace(self.params['mass_mini'], self.params['mass_maxi'], 101)

        self.zred_dist = priors.FastUniform(a=self.params['zred_mini'], b=self.params['zred_maxi'])

        self.logsfr_ratios_dist = priors.FastTruncatedEvenStudentTFreeDeg2(hw=5.0, sig=0.3)

    def __len__(self):
        """Hack to work with Prospector 0.3
        """
        return 9 # logsfr_ratios has 6 bins

    @property
    def range(self):
        return ((self.params['zred_mini'], self.params['zred_maxi']),\
                (self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']),\
                (self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
               )

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x: used to calculate the prior
            x[0] = zred, x[1] = logmass, x[2] = logzsol, x[3:] = logsfr_ratios

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        if x.ndim == 1:
            # doing mcmc; x is [zred, logmass, logzsol]
            p = np.zeros_like(x)
            # p(m)
            p[1] = mass_func_at_z(x[0], x[1], self.params['const_phi'])
            # p(zsol)
            met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                  mu=loc_massmet(x[1]), sig=scale_massmet(x[1]))
            # sfh = sfrd
            logsfr_ratios = expe_logsfr_ratios(x[0], x[1], self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
            p[3:] = t.pdf(x[3:], df=2, loc=logsfr_ratios, scale=0.3)

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(p)

            # p(z)
            lnp[0] = self.zred_dist(x[0])
            lnp[2] = met_dist(x[2]) # FastTruncatedNormal returns ln(p)

            return lnp

        else:
            # write_hdf5. last step.
            # in prior_product, x is of size (nsamples, npriors)
            # Fast* is not vectorized?
            # so just do a loop here
            _zreds = x[...,0]

            all_p = []
            for i in range(len(_zreds)):
                new_x = x[i]
                p = np.zeros_like(new_x)
                p[1] = mass_func_at_z(new_x[0], new_x[1], self.params['const_phi'])
                logsfr_ratios = expe_logsfr_ratios(new_x[0], new_x[1], self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
                p[3:] = t.pdf(new_x[3:], df=2, loc=logsfr_ratios, scale=0.3)

                all_p.append(p)

            all_p = np.array(all_p)

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(all_p)

            met_dists = [priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                    mu=loc_massmet(mass_i), sig=scale_massmet(mass_i)) for mass_i in x[...,1]]
            lnp[...,0] = self.zred_dist(_zreds)
            lnp[...,2] = [met_dists[i](met_i) for (i, met_i) in enumerate(x[...,2])]

            return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.
        Needed for minimizer.

        :param nsample: (optional)
            Unused. Will not work if nsample > 1 in draw_sample()!
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        # draw a zred from pdf(z)
        zred = self.zred_dist.sample()

        # draw from the mass function at the above zred
        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = draw_sample(xs=self.mgrid, cdf=cdf_mass)

        # given mass from above, draw logzsol
        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.sample()

        # sfh = sfrd
        logsfr_ratios = expe_logsfr_ratios(zred, mass, self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
        logsfr_ratios_rvs = t.rvs(df=2, loc=logsfr_ratios, scale=0.3)
        logsfr_ratios_rvs = np.clip(logsfr_ratios_rvs, a_min=self.params['logsfr_ratio_mini'], a_max=self.params['logsfr_ratio_maxi'])

        return np.concatenate([np.atleast_1d(zred), np.atleast_1d(mass),
                               np.atleast_1d(met), np.atleast_1d(logsfr_ratios_rvs)])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        Needed for nested sampling.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        zred = self.zred_dist.unit_transform(x[0])

        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = ppf(x[1], self.mgrid, cdf=cdf_mass)

        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.unit_transform(x[2])

        # sfh = sfrd
        logsfr_ratios = expe_logsfr_ratios(zred, mass, self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])

        logsfr_ratios_ppf = np.zeros_like(logsfr_ratios)
        for i in range(len(logsfr_ratios_ppf)):
            logsfr_ratios_ppf[i] = self.logsfr_ratios_dist.unit_transform(x[3+i]) + logsfr_ratios[i]
        logsfr_ratios_ppf = np.clip(logsfr_ratios_ppf, a_min=self.params['logsfr_ratio_mini'], a_max=self.params['logsfr_ratio_maxi'])
        return np.concatenate([np.atleast_1d(zred), np.atleast_1d(mass),
                               np.atleast_1d(met), np.atleast_1d(logsfr_ratios_ppf)])


##################### p(z)p(logM|z)p(Z*|logM) & SFH(M, z) #####################
# galaxy number density & mass function & Gaussian metallicity & SFH(M, z) priors
# expectation value of the nonparametric SFH ~ Behroozi+19 cosmic SFRD

class NzSFH(priors.Prior):

    prior_params = ['zred_mini', 'zred_maxi', 'mass_mini', 'mass_maxi',
                    'z_mini', 'z_maxi', 'logsfr_ratio_mini', 'logsfr_ratio_maxi',
                    'const_phi'] # mass is in log10

    def __init__(self, parnames=[], name='', **kwargs):
        """Overwrites __init__ in the base code Prior

        Parameters
        ----------
        parnames : sequence of strings
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

        # load stored tables and then interpolate
        # the tables were calculated in pdf_z_tables.ipynb
        # redshift range is 0 - 20
        if self.params['const_phi']:
            zreds, pdf_zred = np.loadtxt(file_pdf_of_z_l20, unpack=True)
        else:
            zreds, pdf_zred = np.loadtxt(file_pdf_of_z_l20t18, unpack=True)

        self.finterp_z_pdf, self.finterp_cdf_z = norm_pz(self.params['zred_mini'], self.params['zred_maxi'], zreds, pdf_zred)
        self.mgrid = np.linspace(self.params['mass_mini'], self.params['mass_maxi'], 101)
        self.zred_dist = priors.FastUniform(a=self.params['zred_mini'], b=self.params['zred_maxi'])
        self.logsfr_ratios_dist = priors.FastTruncatedEvenStudentTFreeDeg2(hw=5.0, sig=0.3)

    def __len__(self):
        """Hack to work with Prospector 0.3
        """
        return 9 # logsfr_ratios has 6 bins

    @property
    def range(self):
        return ((self.params['zred_mini'], self.params['zred_maxi']),\
                (self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']),\
                (self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
               )

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x: used to calculate the prior
            x[0] = zred, x[1] = logmass, x[2] = logzsol, x[3:] = logsfr_ratios

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        if x.ndim == 1:
            # doing mcmc; x is [zred, logmass, logzsol]
            p = np.zeros_like(x)
            # p(z)
            p[0] = self.finterp_z_pdf(x[0])
            # p(m)
            p[1] = mass_func_at_z(x[0], x[1], self.params['const_phi'])
            # p(zsol)
            met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                  mu=loc_massmet(x[1]), sig=scale_massmet(x[1]))
            # sfh = sfrd
            logsfr_ratios = expe_logsfr_ratios(x[0], x[1], self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
            p[3:] = t.pdf(x[3:], df=2, loc=logsfr_ratios, scale=0.3)

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(p)

            lnp[2] = met_dist(x[2]) # FastTruncatedNormal returns ln(p)

            return lnp

        else:
            # write_hdf5. last step.
            # in prior_product, x is of size (nsamples, npriors)
            # Fast* is not vectorized?
            # so just do a loop here
            _zreds = x[...,0]

            all_p = []
            for i in range(len(_zreds)):
                new_x = x[i]
                p = np.zeros_like(new_x)
                p[0] = self.finterp_z_pdf(new_x[0])
                p[1] = mass_func_at_z(new_x[0], new_x[1], self.params['const_phi'])
                logsfr_ratios = expe_logsfr_ratios(new_x[0], new_x[1], self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
                p[3:] = t.pdf(new_x[3:], df=2, loc=logsfr_ratios, scale=0.3)

                all_p.append(p)

            all_p = np.array(all_p)

            with np.errstate(invalid='ignore', divide='ignore'):
                lnp = np.log(all_p)

            met_dists = [priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                                    mu=loc_massmet(mass_i), sig=scale_massmet(mass_i)) for mass_i in x[...,1]]
            lnp[...,2] = [met_dists[i](met_i) for (i, met_i) in enumerate(x[...,2])]

            return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.
        Needed for minimizer.

        :param nsample: (optional)
            Unused. Will not work if nsample > 1 in draw_sample()!
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        # draw a zred from pdf(z)
        u = np.random.uniform(0, 1, size=nsample)
        zred = self.finterp_cdf_z(u)

        # draw from the mass function at the above zred
        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = draw_sample(xs=self.mgrid, cdf=cdf_mass)

        # given mass from above, draw logzsol
        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.sample()

        # sfh = sfrd
        logsfr_ratios = expe_logsfr_ratios(zred, mass, self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
        logsfr_ratios_rvs = t.rvs(df=2, loc=logsfr_ratios, scale=0.3)
        logsfr_ratios_rvs = np.clip(logsfr_ratios_rvs, self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])

        return np.concatenate([np.atleast_1d(zred), np.atleast_1d(mass),
                               np.atleast_1d(met), np.atleast_1d(logsfr_ratios_rvs)])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        Needed for nested sampling.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)

        zred = self.finterp_cdf_z(x[0])

        cdf_mass = cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'])
        mass = ppf(x[1], self.mgrid, cdf=cdf_mass)

        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=loc_massmet(mass), sig=scale_massmet(mass))
        met = met_dist.unit_transform(x[2])

        # sfh = sfrd
        logsfr_ratios = expe_logsfr_ratios(zred, mass, self.params['logsfr_ratio_mini'], self.params['logsfr_ratio_maxi'])
        logsfr_ratios_ppf = np.zeros_like(logsfr_ratios)
        for i in range(len(logsfr_ratios_ppf)):
            logsfr_ratios_ppf[i] = self.logsfr_ratios_dist.unit_transform(x[3+i]) + logsfr_ratios[i]
        logsfr_ratios_ppf = np.clip(logsfr_ratios_ppf, a_min=self.params['logsfr_ratio_mini'], a_max=self.params['logsfr_ratio_maxi'])
        return np.concatenate([np.atleast_1d(zred), np.atleast_1d(mass),
                               np.atleast_1d(met), np.atleast_1d(logsfr_ratios_ppf)])
                               

############################# necessary functions #############################

def scale_massmet(mass):
    """std of the Gaussian approximating the mass-met relationship
    """
    upper_84 = np.interp(mass, massmet[:, 0], massmet[:, 3])
    lower_16 = np.interp(mass, massmet[:, 0], massmet[:, 2])
    return (upper_84-lower_16)

def loc_massmet(mass):
    """mean of the Gaussian approximating the mass-met relationship
    """
    return np.interp(mass, massmet[:, 0], massmet[:, 1])

############ Mass function in Leja+20    ############
############ Code modified from appendix ############
def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """
    Generate a Schechter function (in dlogm).
    """
    phi = ((10**logphi) * np.log(10) *
           10**((logm - logmstar) * (alpha + 1)) *
           np.exp(-10**(logm - logmstar)))
    return phi

def parameter_at_z0(y,z0,z1=0.2,z2=1.6,z3=3.0):
    """
    Compute parameter at redshift ‘z0‘ as a function
    of the polynomial parameters ‘y‘ and the
    redshift anchor points ‘z1‘, ‘z2‘, and ‘z3‘.
    """
    y1, y2, y3 = y
    a = (((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) /
         (z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)))
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c

# Continuity model median parameters + 1-sigma uncertainties.
pars = {'logphi1': [-2.44, -3.08, -4.14],
        'logphi1_err': [0.02, 0.03, 0.1],
        'logphi2': [-2.89, -3.29, -3.51],
        'logphi2_err': [0.04, 0.03, 0.03],
        'logmstar': [10.79,10.88,10.84],
        'logmstar_err': [0.02, 0.02, 0.04],
        'alpha1': [-0.28],
        'alpha1_err': [0.07],
        'alpha2': [-1.48],
        'alpha2_err': [0.1]}

def draw_at_z(z0=1.0):
    '''The Leja+20 mass function is only defined over 0.2<=z<=3.
    If 'z0' is outside the range, we use the z=0.2 and z=3 parameter values
    in this function.
    '''
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

    for par in ['logphi1', 'logphi2', 'logmstar', 'alpha1', 'alpha2']:
        samp = pars[par]
        if par in ['logphi1', 'logphi2', 'logmstar']:
            draws[par] = parameter_at_z0(samp,_z0)
        else:
            draws[par] = np.array(samp)

    return draws

def low_z_mass_func(z0, logm):
    '''Mass function in Leja+20
    logm: an array of [mass_mini, ..., mass_maxi], or a float.
    returns: an array of phi as a function of logm, or phi at z0.
    '''
    draws = draw_at_z(z0=z0)
    phi1 = schechter(logm, draws['logphi1'],  # primary component
                     draws['logmstar'], draws['alpha1'])
    phi2 = schechter(logm, draws['logphi2'],  # secondary component
                     draws['logmstar'], draws['alpha2'])

    phi = phi1 + phi2

    return np.squeeze(phi)


############ Mass function in Tacchella+18 ############
def scale(x, zrange=[3, 4], trigrange=[0, np.pi/2]):

    x_std = (x - zrange[0]) / (zrange[1] - zrange[0])
    x_scaled = x_std * (trigrange[1] - trigrange[0]) + trigrange[0]

    return x_scaled

def cos_weight(z, zrange=[3, 4]):
    '''cos^2 weighted average

    zrange: min and max of the redshift range over which we will take the cos^2 weighted average
    '''

    z_scaled = scale(z, zrange=zrange)
    w_l20 = np.cos(z_scaled)**2
    w_t18 = 1 - np.cos(z_scaled)**2

    return np.array([w_l20, w_t18])

# Best-fit parameters from table 2.
z_t18 = np.arange(4, 13, 1)
phi_t18 = np.array([261.9, 201.2, 140.5, 78.0, 38.4, 37.3, 8.1, 3.9, 1.1])
phi_t18 *= 1e-5
logm_t18 = np.array([10.16, 9.89, 9.62, 9.38, 9.18, 8.74, 8.79, 8.50, 8.50])
m_t18 = 10**logm_t18
alpha_t18 = np.array([-1.54, -1.59, -1.64, -1.70, -1.76, -1.80, -1.92, -2.00, -2.10])

def schechter_t18(m, phi_star, m_star, alpha):
    '''In linear space
    '''
    return phi_star*(m/m_star)**(alpha+1) * np.exp(-m/m_star)

def high_z_mass_func_discreate(z0, this_m):
    # no boundary check is done here
    idx = np.squeeze(np.where(z_t18==int(np.round(z0))))
    return schechter_t18(m=this_m, phi_star=phi_t18[idx], m_star=m_t18[idx], alpha=alpha_t18[idx])

def high_z_mass_func(z0, this_m):
    '''Take cos^2 weighted average in each bin, with zrange being the bin edges.
    '''
    if z0 <= 4.0:
        return high_z_mass_func_discreate(4.0, this_m)
    elif z0 >= 12.0:
        return high_z_mass_func_discreate(12.0, this_m)
    else:
        zrange = [int(z0), int(z0)+1]

        phi0 = high_z_mass_func_discreate(zrange[0], this_m)
        phi1 = high_z_mass_func_discreate(zrange[1], this_m)

        w = cos_weight(z0, zrange=zrange)
        phi = phi0*w[0] + phi1*w[1]
        return phi

def mass_func_at_z(z, this_logm, const_phi=False):
    '''
    if const_phi == True: use mass funtions in Leja+20 only;
                          no redshfit evolution outside the range 0.2 <= z <= 3.0;
        i.e.,
        z<=0.2: use mass funtion at z=0.2;
        0.2<=z<=3.0: defined in Leja+20;
        z>=3: use mass funtion at z=3.0.
    if const_phi == False: combine Leja+20 and Tacchella+18 mass functions;
        i.e.,
        z<=3: mass function in Leja+20; continuous in redshift.
        3<z<4: cos^2 weighted average of L20 mass function at z=3 and T18 mass function at z=4;
               i.e., we weight the L20 mass funtion as cos^2(0, pi/2), and T18 as 1 - cos^2(0, pi/2).
        4<=z<=12: T18 mass function; discreate in redshift by definition;
                  we take cos^2 weighted average in each bin, with zrange being the bin edges.
        z>12: T18 mass function at z=12.
    '''
    if const_phi:
        phi = low_z_mass_func(z, this_logm)
    else:
        if z <= 3.0:
            phi = low_z_mass_func(z0=z, logm=this_logm)
        elif z > 3.0 and z < 4.0:
            phi_lowz = low_z_mass_func(z0=3, logm=this_logm)
            phi_highz = high_z_mass_func(z0=4, this_m=10**this_logm)
            w = cos_weight(z)
            phi = phi_lowz*w[0] + phi_highz*w[1]
        elif z >= 4.0 and z <= 12.0:
            phi = high_z_mass_func(z0=z, this_m=10**this_logm)
        else:
            phi = high_z_mass_func(z0=12, this_m=10**this_logm)

    return np.squeeze(phi)


############ Empirical PDF & CDF ############
def pdf_mass_func_at_z(z, logm, const_phi):
    phi_50 = mass_func_at_z(z, logm, const_phi)
    p_phi_int = np.trapz(phi_50, logm)
    pdf_at_m = phi_50/p_phi_int
    return pdf_at_m

def cdf_mass_func_at_z(z, logm, const_phi):
    '''
    logm: an array of [mass_mini, ..., mass_maxi], or a float
    '''
    pdf_at_m = pdf_mass_func_at_z(z, logm=logm, const_phi=const_phi)
    cdf_of_m = np.cumsum(pdf_at_m)
    cdf_of_m /= max(cdf_of_m)

    # may have small numerical errors; force cdf to start at 0
    clean = np.where(cdf_of_m < 0)
    cdf_of_m[clean] = 0
    cdf_of_m[0] = 0

    return cdf_of_m

def ppf(x, xs, cdf):
    '''Go from a value x of the CDF (between 0 and 1) to
    the corresponding parameter value.
    '''
    func_interp = interp1d(cdf, xs, bounds_error=False, fill_value="extrapolate")
    param = func_interp(x)
    return param

def draw_sample(xs, cdf, nsample=None):
    '''Draw sample(s) from any cdf
    '''
    u = np.random.uniform(0, 1, size=nsample)
    func_interp = interp1d(cdf, xs, bounds_error=False, fill_value="extrapolate")
    sample = func_interp(u)
    return sample

def norm_pz(zred_mini, zred_maxi, zreds, pdf_zred):
    """normalize int_{zred_mini}^{zred_maxi} p(z) = 1
    """
    idx_zrange = np.logical_and(zreds>=zred_mini, zreds<=zred_maxi)
    zreds_inrange = zreds[idx_zrange]
    p_int = np.trapz(pdf_zred[idx_zrange], zreds_inrange)
    pdf_zred_inrange = pdf_zred[idx_zrange]/p_int
    invalid = np.where(pdf_zred_inrange<0)
    pdf_zred_inrange[invalid] = 0
    finterp_z_pdf = interp1d(zreds_inrange, pdf_zred_inrange, bounds_error=False, fill_value="extrapolate")

    cdf_zred = np.cumsum(pdf_zred_inrange)
    cdf_zred /= max(cdf_zred)
    invalid = np.where(cdf_zred<0)
    cdf_zred[invalid] = 0
    finterp_cdf_z = interp1d(cdf_zred, zreds_inrange, bounds_error=False, fill_value="extrapolate")

    return (finterp_z_pdf, finterp_cdf_z)


############ Needed for SFH(M,z) ############
def z_to_agebins_rescale(zobs, agelims=np.array([0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0]), amin=7.1295):
    """agelims here must match those in z_to_agebins(), which set the nonparameteric
    SFH age bins depending on the age of the universe at a given z.

    This function ensures that the agebins used for calculating the expectation values of logsfr_ratios
    follow the same spacing.
    """

    agelims[0] = cosmos.lookback_time(zobs).to(u.yr).value # shift the start to z_obs

    tuniv = 13768899116.929323 # cosmos.age(0).value*1e9
    tbinmax = tuniv - (tuniv-agelims[0]) * 0.10
    agelims[-2] = tbinmax
    agelims[-1] = tuniv

    if zobs <= 3.0:
        agelims[1] = agelims[0] + 3e7 # 1st bin is 30 Myr wide
        agelims[2] = agelims[1] + 1e8 # 2nd bin is 100 Myr wide
        i_age = 3
        nbins = 5
    else:
        agelims[1] = agelims[0] + 10**amin
        i_age = 2
        nbins = 6

    if agelims[0] == 0:
        with np.errstate(invalid='ignore', divide='ignore'):
            agelims = np.log10(agelims[:i_age]).tolist()[:-1] + np.squeeze(np.linspace(np.log10(agelims[i_age-1]),np.log10(tbinmax),nbins)).tolist() + [np.log10(tuniv)]
            agelims[0] = 0

    else:
        agelims = np.log10(agelims[:i_age]).tolist()[:-1] + np.squeeze(np.linspace(np.log10(agelims[i_age-1]),np.log10(tbinmax),nbins)).tolist() + [np.log10(tuniv)]

    return (np.array([agelims[:-1], agelims[1:]]).T)

def slope(x, y):
    return (y[1]-y[0])/(x[1]-x[0])

def slope_and_intercept(x, y):
    a = slope(x, y)
    b = -x[0]*a + y[0]
    return a, b

def delta_t_dex(m, mlims=[9, 12], dlims=[-0.6, 0.4]):
    """Introduces mass dependence in SFH
    """
    a, b = slope_and_intercept([mlims[0], mlims[1]], [dlims[0], dlims[1]])

    if m <= mlims[0]:
        return dlims[0]
    elif m >= mlims[1]:
        return dlims[1]
    else:
        return a * m + b

def expe_logsfr_ratios(this_z, this_m, logsfr_ratio_mini, logsfr_ratio_maxi):
    """expectation values of logsfr_ratios
    """

    age_shifted = np.log10(cosmos.age(this_z).value) + delta_t_dex(this_m)
    age_shifted = 10**age_shifted

    zmin_thres = 1e-4
    zmax_thres = 20
    if age_shifted < age[-1]:
        z_shifted = zmax_thres * 1
    elif age_shifted > age[0]:
        z_shifted = zmin_thres * 1
    else:
        z_shifted = f_age_z(age_shifted)
        if z_shifted > zmax_thres:
            z_shifted = zmax_thres * 1

    agebins_in_yr_rescaled_shifted = z_to_agebins_rescale(z_shifted)
    agebins_in_yr_rescaled_shifted = 10**agebins_in_yr_rescaled_shifted
    agebins_in_yr_rescaled_shifted_ctr = np.mean(agebins_in_yr_rescaled_shifted, axis=1)

    nsfrbins = agebins_in_yr_rescaled_shifted.shape[0]

    sfr_shifted = np.zeros(nsfrbins)
    sfr_shifted_ctr = np.zeros(nsfrbins)
    for i in range(nsfrbins):
        a = agebins_in_yr_rescaled_shifted[i,0]
        b = agebins_in_yr_rescaled_shifted[i,1]
        sfr_shifted[i] = spl_tl_sfrd.integral(a=a, b=b)
        sfr_shifted_ctr[i] = spl_tl_sfrd(agebins_in_yr_rescaled_shifted_ctr[i])

    logsfr_ratios_shifted = np.zeros(nsfrbins-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(nsfrbins-1):
            logsfr_ratios_shifted[i] = np.log10(sfr_shifted[i]/sfr_shifted[i+1])
    logsfr_ratios_shifted = np.clip(logsfr_ratios_shifted, logsfr_ratio_mini, logsfr_ratio_maxi)

    if not np.all(np.isfinite(logsfr_ratios_shifted)):
        # set nan accord. to its neighbor
        nan_idx = np.isnan(logsfr_ratios_shifted)
        finite_idx = np.min(np.where(nan_idx==True))-1
        neigh = logsfr_ratios_shifted[finite_idx]
        nan_idx = np.arange(6-finite_idx-1) + finite_idx + 1
        for i in range(len(nan_idx)):
            logsfr_ratios_shifted[nan_idx[i]] = neigh * 1.

    return logsfr_ratios_shifted
