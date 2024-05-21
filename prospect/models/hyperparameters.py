"""
hyperparameters.py


This class gets all all the ProspectorParams functionality, but it overrides the
_prior_product and prior_transform methods to sample the hyperparameters & log
SFR ratios of the stochastic SFH prior.
"""

import numpy as np
import scipy
from . import priors
from . import hyperparam_transforms as transforms
from .parameters import ProspectorParams

__all__ = ["ProspectorHyperParams"]


class ProspectorHyperParams(ProspectorParams):

    """
    This class implements a SFH prior that is determined by hyper-parameters
    that in turn have their own prior distributions.
    """

    def _prior_product(self, theta, **extras):
        """Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the prior
        functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values. ndarray of
            shape ``(ndim,)``

        :returns lnp_prior:
            The natural log of the product of the prior probabilities for these
            parameter values.
        """
        lnp_prior = 0

        hyper_params = ['sigma_reg', 'tau_eq', 'tau_in', 'sigma_dyn', 'tau_dyn']
        psd_params = np.zeros(len(hyper_params))

        for i, p in enumerate(hyper_params):
            if self.config_dict[p]['isfree']:
                inds = self.theta_index[p]
                psd_params[i] = theta[..., inds][0]
                func = self.config_dict[p]['prior']
                this_prior = np.sum(func(theta[..., inds]), axis=-1)
                lnp_prior += this_prior
            else:
                psd_params[i] = self.config_dict[p]['init']

        sfr_covar_matrix = transforms.get_sfr_covar(psd_params, agebins=self.config_dict['agebins']['init'])
        sfr_ratio_covar_matrix = transforms.sfr_covar_to_sfr_ratio_covar(sfr_covar_matrix)
        nbins = len(self.config_dict['agebins']['init'])
        logsfr_ratio_prior = scipy.stats.multivariate_normal(mean=[0.]*(nbins-1), cov=sfr_ratio_covar_matrix)
        inds = self.theta_index['logsfr_ratios']
        this_prior = np.sum(np.log(logsfr_ratio_prior.pdf(theta[..., inds])))
        lnp_prior += this_prior

        for k, inds in list(self.theta_index.items()):
            if (k in hyper_params) or (k == 'logsfr_ratios'):
                continue
            func = self.config_dict[k]['prior']
            this_prior = np.sum(func(theta[..., inds]), axis=-1)
            lnp_prior += this_prior

        return lnp_prior


    def prior_transform(self, unit_coords):
        """Go from unit cube to parameter space, for nested sampling.

        :param unit_coords:
            Coordinates in the unit hyper-cube. ndarray of shape ``(ndim,)``.

        :returns theta:
            The parameter vector corresponding to the location in prior CDF
            corresponding to ``unit_coords``. ndarray of shape ``(ndim,)``
        """

        theta = np.zeros(len(unit_coords))

        hyper_params = ['sigma_reg', 'tau_eq', 'tau_in', 'sigma_dyn', 'tau_dyn']
        psd_params = np.zeros(len(hyper_params))


        for i, p in enumerate(hyper_params):
            if self.config_dict[p]['isfree']:
                func = self.config_dict[p]['prior'].unit_transform
                inds = self.theta_index[p]
                psd_params[i] = func(unit_coords[inds])
                theta[inds] = psd_params[i]
            else:
                psd_params[i] = self.config_dict[p]['init']

        sfr_covar_matrix = transforms.get_sfr_covar(psd_params, agebins=self.config_dict['agebins']['init'])
        sfr_ratio_covar_matrix = transforms.sfr_covar_to_sfr_ratio_covar(sfr_covar_matrix)
        logsfr_ratio_prior = priors.MultiVariateNormal(mean=0, Sigma=sfr_ratio_covar_matrix)
        x = unit_coords[self.theta_index['logsfr_ratios']]
        logsfr_ratios = logsfr_ratio_prior.unit_transform(x)
        theta[self.theta_index['logsfr_ratios']] = logsfr_ratios

        for k, inds in list(self.theta_index.items()):
            if (k in hyper_params) or (k == 'logsfr_ratios'):
                continue
            func = self.config_dict[k]['prior'].unit_transform
            theta[inds] = func(unit_coords[inds])

        return theta




