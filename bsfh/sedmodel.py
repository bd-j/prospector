import numpy as np
from scipy.interpolate import interp1d
from bsfh.parameters import ProspectrParams
try:
    from astropy.cosmology import WMAP9 as cosmo
except(ImportError):
    pass

lsun = 3.846e33  # ergs/s
pc = 3.085677581467192e18  # cm
jansky_mks = 1e-26
#value to go from L_sun/Hz to erg/s/cm^2/Hz at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )

class SedModel(ProspectrParams):
    """
    For models composed of SSPs and sums of SSPs which use the
    sps_basis.StellarPopBasis as the sps object.
    """

    def mean_model(self, theta, obs, sps=None, **extras):
        """
        Given a theta vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        :param theta:
            ndarray of parameter values.

        :param obs:
            An observation dictionary, containing the output
            wavelength array, the photometric filter lists, and the
            key 'logify_spectrum' which is True if the comparison to
            the model is to be made in the log.
            
        :param sps:
            A StellarPopBasis object to be used
            in the model generation.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by obs['wavelength'], and optionally in the log.
            
        :returns phot:
            The model photometry for these parameters, for the filters
            specified in obs['filters'].
            
        :returns extras:
            Any extra aspects of the model that are returned.
        """
        s, p, x = self.sed(theta, obs, sps=sps, **extras)
        if obs.get('logify_spectrum', True):
            s = np.log(s) + np.log(self.spec_calibration(obs=obs, **extras))
        else:
            s *= self.spec_calibration(obs=obs, **extras)
        return s, p, x
    
    def sed(self, theta, obs, sps=None, **kwargs):
        """
        Given a theta vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), ***not** including any instrument
        calibration effects.

        :param theta:
            ndarray of parameter values.
            
        :param sps:
            A StellarPopBasis object to be used
            in the model generation.

        :returns spec:
            The model spectrum for these parameters, at the wavelengths
            specified by obs['wavelength'], in linear units.
            
        :returns phot:
            The model photometry for these parameters, for the filters
            specified in obs['filters'].
            
        :returns extras:
            Any extra aspects of the model that are returned.
        """
        
        self.set_parameters(theta)        
        spec, phot, extras = sps.get_spectrum(outwave=obs['wavelength'],
                                              filters=obs['filters'],
                                              **self.params)
        
        spec *= obs.get('normalization_guess', 1.0)
        #remove negative fluxes
        try:
            tiny = 1.0/len(spec) * spec[spec > 0].min()
            spec[ spec < tiny ] = tiny
        except:
            pass
        spec = (spec + self.sky())
        return spec, phot, extras

    def sky(self):
        """Model for the *additive* sky emission/absorption"""
        return 0.
        
    def spec_calibration(self, theta=None, obs=None, **kwargs):
        """
        Implements a polynomial calibration model.  This only happens
        if `pivot_wave` is a key of the obs dictionary, since the
        polynomial is defined in terms of r'$x \equiv
        \lambda/\lambda_{{pivot}} - 1$'.

        :returns cal:
           a polynomial given by 'spec_norm' * (1 + \Sum_{m=1}^M
           'poly_coeffs'[m-1] x**m)
        """
        if theta is not None:
            self.set_parameters(theta)
        
        # Should find a way to make this more generic,
        # using chebyshev polynomials
        if 'pivot_wave' in obs:
            x = obs['wavelength']/obs['pivot_wave'] - 1.0
            poly = np.zeros_like(x)
            powers = np.arange( len(self.params['poly_coeffs']) ) + 1
            poly = (x[None,:] ** powers[:,None] *
                    self.params['poly_coeffs'][:,None]).sum(axis = 0)
            #switch to have spec_norm be multiplicative or additive
            #depending on whether the calibration model is
            #multiplicative in exp^poly or just poly, Should move this
            #to mean_model()?
            if self.params.get('cal_type', 'exp_poly') is 'poly':
                return (1.0 + poly) * self.params['spec_norm']
            else:
                return np.exp(self.params['spec_norm'] + poly)
        else:
            return 1.0

    def spec_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        pars = ['gp_jitter', 'gp_amplitude', 'gp_length']
        defaults = [[0.0], [0.0], [1.0]]
        vals = [self.params.get(p, d) for p, d in zip(pars, defaults)]
        return  tuple(vals)
    
    def phot_gp_params(self, theta=None, **extras):
        if theta is not None:
            self.set_parameters(theta)
        s = self.params.get('phot_jitter', 0.0)
        return s, [0.0], [0]

    
class CSPModel(ProspectrParams):
    """
    For parameterized SFHs where fsps.StellarPopulation is used as the
    sps object.
    """

    def mean_model(self, theta, obs, sps=None, **kwargs):
        """Rename of CSPModel.sed() for compatibility.  If any
        parameteric calibration model is applied, it should go here.
        """
        return self.sed(theta, obs, sps=sps, **kwargs)
    
    def sed(self, theta, obs, sps=None, **kwargs):
        """
        Given a theta vector, generate spectroscopy, photometry and any
        extras (e.g. stellar mass).

        :param theta:
            ndarray of parameter values.

        :param sps:
            A python-fsps StellarPopulation object to be used for
            generating the SED.

        :returns spec:
            The restframe spectrum in units of L_\odot/Hz
            
        :returns phot:
            The apparent (redshifted) maggies in each of the filters.

        :returns extras:
            A list of None type objects, only included for consistency
            with the SedModel class.
        """
        self.set_parameters(theta)
        # Pass the model parameters through to the sps object
        ncomp = len(self.params['mass'])
        for ic in range(ncomp):
            s, p, x = self.one_sed(component_index = ic, sps=sps,
                                   filterlist=obs['filters'])
            try:
                spec += s
                maggies += p
                extra += [x]
            except(NameError):
                spec, maggies, extra = s, p, [x]
                
        if obs['wavelength'] is not None:
            spec = interp1d( sps.wavelengths, spec, axis = -1,
                             bounds_error=False)(obs['wavelength'])
                
        #modern FSPS does the distance modulus for us in get_mags,
        # !but python-FSPS does not!
        if sps.params['zred'] == 0:
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('dist', 1e-5) * 1e5)**2
            # spectrum stays in L_sun/Hz
            dfactor_spec = 1.0
        else:
            dfactor = ((cosmo.luminosity_distance(sps.params['zred']).value *
                        1e5)**2 / (1+sps.params['zred']))
            # convert to maggies
            dfactor_spec = to_cgs / 1e3 / dfactor / (3631*jansky_mks)
        return (spec * dfactor_spec + self.sky(),
                maggies / dfactor,
                extra)

    def one_sed(self, component_index=0, sps=None, filterlist=[]):
        """Get the SED of one component for a multicomponent composite
        SFH.  Should set this up to work as an iterator.
        
        :param component_index:
            Integer index of the component to calculate the SED for.
            
        :params sps:
            A python-fsps StellarPopulation object to be used for
            generating the SED.

        :param filterlist:
            A list of strings giving the (FSPS) names of the filters
            onto which the spectrum will be projected.
            
        :returns spec:
            The restframe spectrum in units of L_\odot/Hz.
            
        :returns maggies:
            Broadband fluxes through the filters named in
            ``filterlist``, ndarray.  Units are observed frame
            absolute maggies: M = -2.5 * log_{10}(maggies).
            
        :returns extra:
            The extra information corresponding to this component.
        """
        # Pass the model parameters through to the sps object,
        # and keep track of the mass of this component
        mass = 1.0
        for k, vs in self.params.iteritems():
            try:
                v = vs[component_index]
                #n_param_is_vec += 1
            except(IndexError, TypeError):
                v = vs
            if k in sps.params.all_params:
                if k == 'zmet':
                    vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
                else:
                    vv = v.copy()
                sps.params[k] = vv
            if k == 'mass':
                mass = v
        #now get the magnitudes and spectrum
        w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
        mags = sps.get_mags(tage=sps.params['tage'],
                            bands=filterlist)
        # normalize by (current) stellar mass and get correct units (distance_modulus)
        mass_norm = mass/sps.stellar_mass
        return (mass_norm * spec,
                mass_norm * 10**(-0.4*(mags)),
                sps.stellar_mass)

    def phot_calibration(self, **extras):
        return 1.0

    def phot_gp_params(self, obs=None, theta=None,
                       **extras):
        """Return the parameters for generating the covariance matrix
        used by the photometric gaussian process in a way
        understandable for the GP objects.  This method looks for the
        ``phot_jitter`` parameter.  For the outlier modeling it also
        looks for  ``gp_outlier_amps`` and ``gp_outlier_locs`` keys in
        the parameter dictionary.  For the additional error on grouped
        bands it looks for ``gp_filter_locs`` and ``gp_filter_amps``
        keys in the parameters dictionary.

        The ``gp_outlier_locs`` is an array of indices into the
        (masked) maggies array, and ``gp_outlier_amps`` is an array of
        GP amplitudes corresponding to these indices.

        Otherwise, ``gp_filter_locs`` is a fixed array of filter name
        lists (or other iterable), and ``gp_filter_amps`` is an array
        of the same length giving the GP amplitudes associated with
        each list

        :param obs:
            obs data dictionary.  Must have a ``filters`` key.  If not
            supplied, only the overall jitter term will be searched
            and amps and locs will be returned as [0], [0].
            
        :param theta:
            Theta parameter vector.  If suppied, parameters will be
            set before being parsed into s, amps, locs
            
        :returns s:
            The jitter (scalar)

        :returns amps:
            The GP amplitudes, ndarray.

        :returns locs:
            Indices into the obs['maggies'][obs['phot_mask']] array
            corresponding elementwise to the returned ``amps`` array
        """

        if theta is not None:
            self.set_parameters(theta)
            
        # Overall jitter
        s = self.params.get('phot_jitter', 0.0)
        if obs is None:
            return s, [0.0], [0]
        
        # band dependent jitter
        mask = obs.get('phot_mask', np.ones( len(obs['filters']), dtype= bool))
        noise = np.zeros(len(mask))
        if 'str' in str(type(obs['filters'][0])):
            fnames = [f for i,f in enumerate(obs['filters']) if mask[i]]
        else:
            fnames = [f.name for i,f in enumerate(obs['filters']) if mask[i]]

        # Do the outlier modeling
        outl = self.params.get('gp_outlier_locs', np.array([0]))
        outa = self.params.get('gp_outlier_amps', np.array([0]))
        allowed_names = self.params.get('gp_outlier_allowed', fnames)
        if (outl.any()) and (outa.any()):
            # Outlier modeling allows the locations to float. locs here are indices
            ainds =  np.array([fnames.index(f) for f in allowed_names])
            outl = np.clip(outl, 0, len(allowed_inds) - 1).astype(int)
            noise[ainds[outl]] = outa**2

        # Here we add linked amplitudes based on filter names.  locs
        # here is an array of lists
        locs = self.params.get('gp_filter_locs', np.array([0]))
        amps = self.params.get('gp_filter_amps', np.array([0]))
        if (not locs.any()) and (not amps.any()):
            return s, np.atleast_1d(outa), np.atleast_1d(outl)
        
        ll, aa = [], []
        for i, a in enumerate(amps):
            filts = locs[i]
            ll += [fnames.index(f) for f in filts if f in fnames]
            aa += len(filts) * [a]
        noise[ll] += np.array(aa)**2
        ll = noise > 0
        aa = np.sqrt(noise[ll])

        return  s, aa, np.where(ll)[0]

    def sky(self):
        return 0.


def gauss(x, mu, A, sigma):
    """
    Sample multiple gaussians at positions x.

    :param x:
        locations where samples are desired.

    :param mu:
        Center(s) of the gaussians.
        
    :param A:
        Amplitude(s) of the gaussians, defined in terms of total area.

    :param sigma:
        Dispersion(s) of the gaussians, un units of x.

    :returns val:
        The values of the sum of gaussians at x 
    """ 
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)


class HMCThetaParameters(SedModel):
    """
    Object describing a model parameter set, and conversions between a
    parameter dictionary and a theta vector (for use in MCMC sampling).
    Also contains a method for computing the prior probability of a given
    theta vector.
    """

    def lnp_prior_grad(self, theta):
        """
        Return a vector of gradients in the prior probability.
        Requires  that functions giving the gradients are given in the
        theta descriptor.

        :param theta:
            A theta parameter vector containing the desired
            parameters.  ndarray of shape (ndim,)
        """
        lnp_prior_grad = np.zeros_like(theta)
        for k, v in self.theta_index.iteritems():
            start, stop =v
            lnp_prior_grad[start:stop] = (self._config_dict[k]['prior_gradient_function']
                                          (theta[start:stop],
                                           **self._config_dict[k]['prior_args']))
        return lnp_prior_grad

    
    def check_constrained(self, theta):
        """
        For HMC, check if the trajectory has hit a wall in any
        parameter.   If so, reflect the momentum and update the
        parameter position in the  opposite direction until the
        parameter is within the bounds. Bounds  are specified via the
        'upper' and 'lower' keys of the theta descriptor.

        :param theta:
            A theta parameter vector containing the desired
            parameters.  ndarray of shape (ndim,)
        """
        oob = True
        sign = np.ones_like(theta)
        if self.verbose: print('theta in={0}'.format(theta))
        while oob:
            oob = False
            for k,v in self.theta_index.iteritems():
                start, end = v
                par = self._config_dict[k]
                if 'upper' in par.keys():
                    above = theta[start:end] > par['upper']
                    oob = oob or np.any(above)
                    theta[start:end][above] = 2 * par['upper'] - theta[start:end][above]
                    sign[start:end][above] *= -1
                if 'lower' in par.keys():
                    below = theta[start:end] < par['lower']
                    oob = oob or np.any(below)
                    theta[start:end][below] = 2 * par['lower'] - theta[start:end][below]
                    sign[start:end][below] *= -1
        if self.verbose: print('theta out={0}'.format(theta))            
        return theta, sign, oob


