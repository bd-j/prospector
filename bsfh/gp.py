import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy import sparse

class GaussianProcess(object):

    def __init__(self, wave=None, sigma=None, kernel=None, flux=1, **extras):
        """
        Initialize the relevant parameters for the gaussian process.

        :param wave:
            The wavelength scale of the points for which you want
            estimates.
           
        :param sigma:
            The uncertainty estimate at each wavelength point.

        :param flux:
            If supplied, the additional noise given by the jitter can
            be specified as a fraction of the flux
        """
        self.reset()
        
        if kernel is None:
            npar = self.kernel_properties[0]
            self.kernel = np.zeros(npar)
        else:
            self.kernel = kernel
        #_params stores the values of kernel parameters used to
        #construct and compute the factorized covariance matrix that
        #is stored in factorized_Sigma
        self.params_clean = False
        self._params = None
        self.data_clean = False
        self.update_data(wave, sigma, flux)
        
        
    def reset(self):
        """Blank out the cached values.
        """
        self.factorized_Sigma = None
        self._wave = None
        self._sigma = None
        self._flux = 1
        self._params = None
        self.kernel = None
        self.data_clean = False
        self.params_clean = False
        
    def update_params(self):
        """Update the internel kernel parameters using values stored
        in the ``kernel`` property, which may be changed explicitly
        from the outside.  If the contents of ``kernel`` are different
        than the chached parameters, set the params_clean flag to
        False.
        """
        params = self.kernel_to_params(self.kernel)
        self.params_clean = np.array_equal(params, self._params)
        self._params = params
        
    def update_data(self, wave, sigma, flux):
        """Update the data used to generate the covariance matrix.  If
        the data has changed, set the data_clean property to False.
        If supplied data are None, use cached values.  Otherwise cache
        the supplied values.
        """
        data_clean = True
        if wave is not None:
            data_clean = data_clean & np.array_equal(wave, self._wave)
            self._wave = wave
        if sigma is not None:
            data_clean = data_clean & np.array_equal(sigma, self._sigma)
            self._sigma = sigma  
        if flux is not None:
            data_clean = data_clean & np.array_equal(flux, self._flux)
            self._flux = flux
        self.data_clean = self.data_clean & data_clean
            
    def compute(self, wave=None, sigma=None, flux=None,
                check_finite=False, force=False, **extras):
        """Construct the covariance matrix, factorize it, and store
        the factorized matrix.  The factorization is only performed if
        the kernel parameters have chenged or the observational data
        (wave and sigma) have changed.
        
        :param wave: optional
            independent vari able.
            
        :param sigma:
            uncertainties on the dependent variable at the locations
            of the independent variable
            
        :param flux:
            A scaling vector (or scalar) for the uncertainties.  A
            value of ``1`` does not scale the uncertainties at all
            
        :param force: optional
            If true, force a recomputation even if the kernel and the
            data are the same as for the stored factorization.
        """
        self.update_params()
        self.update_data(wave, sigma, flux)
        
        if self.params_clean and self.data_clean and (not force):
            # Nothing changed
            return
        
        else:
            # Something changed or we're forcing regeneration
            Sigma = self.construct_covariance()
            self.factorized_Sigma  = cho_factor(Sigma, overwrite_a=True,
                                                check_finite=check_finite)
            self.log_det = 2 * np.sum( np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)
            self.data_clean = True
            self.params_clean = True
            
    def lnlikelihood(self, residual, check_finite=False, **extras):
        """
        Compute the ln of the likelihood, using the current factorized
        covariance matrix.
        
        :param residual: ndarray, shape (nwave,)
            Vector of residuals (y_data - mean_model).
        """
        assert ( len(residual) == len(self._sigma) )
        self.compute()
        first_term = np.dot(residual,
                            cho_solve(self.factorized_Sigma,
                                      residual, check_finite = check_finite))
        lnL = -0.5 * (first_term + self.log_det)
        
        return lnL
              
    def predict(self, residual, wave=None):
        """
        For a given residual vector, give the GP mean prediction at
        each wavelength and the covariance matrix.  This is currently broken.

        :param residual:
            Vector of residuals (y_data - mean_model).
            
        :param wave: default None
            Wavelengths at which mean and variance estimates are desired.
            Defaults to the input wavelengths.
        """
        
        
        Sigma_cross = self.construct_covariance(inwave=wave, cross=True)
        Sigma_star = self.construct_covariance(inwave=wave, cross=False)
        
        mu = np.dot(Sigma_cross, cho_solve(self.factorized_Sigma, residual))
        cov = Sigma_star - np.dot(Sigma_cross, cho_solve(self.factorized_Sigma,
                                                         -Sigma_cross.T))
        return mu, cov

    @property
    def kernel_properties(self):
        """Return a list of kernel properties, where the first element
        is the number of kernel parameters
        """
        raise NotImplementedError
    
    def kernel_to_params(self, kernel):
        """A method that takes an ndarray and returns a blob of kernel
        parameters.  mostly usied for a sort of documentation, but
        also for grouping parameters
        """
        raise NotImplementedError
    
    def construct_covariance(self, inwave=None, cross=False):
        raise NotImplementedError

class ExpSquared(GaussianProcess):

    @property
    def kernel_properties(self):
        return [3]
                              
    def kernel_to_params(self, kernel):
        """Kernel is a vector consisting of log(s, a**2, l**2)
        """
        s, asquared, lsquared = np.exp(kernel).tolist()
        return s, asquared, lsquared

    def construct_covariance(self, inwave=None, cross=False, **extras):
        """Construct an exponential squared covariance matrix
        """
        s, asq, lsq = self._params
            
        if inwave is None:
            Sigma = asq * np.exp(-(self._wave[:,None] - self._wave[None,:])**2/(2*lsq))
            dinds = np.diag_indices_from(Sigma)
            if np.any(self._flux != 1):
                scale = sparse.diags(self._flux, 0)
                Sigma = scale.T.dot(scale.dot(Sigma).T).T
            Sigma[dinds] += (self._sigma**2 + s**2)
            return Sigma
        elif cross:
            Sigma = asq * np.exp(-(inwave[:,None] - self._wave[None,:])**2/(2*lsq))
            return Sigma
        else:
            Sigma = asq * np.exp(-(inwave[:,None] - inwave[None,:])**2/(2*lsq))
            Sigma[np.diag_indices_from(Sigma)] += s**2
            return Sigma
        
class PhotOutlier(GaussianProcess):

    @property
    def kernel_properties(self):
        return [3]
    
    def kernel_to_params(self, kernel):
        """Kernel is a set of (diagonal) locations and amplitudes.
        The last element is the jitter term.
        """
        jitter = int((len(kernel) % 2 == 1))
        nout = (len(kernel)-jitter) / 2
        amps = kernel[:nout]
        locs = kernel[nout:2*nout]
        if jitter:
            jitter = kernel[-1]
        return jitter, locs, amps

    def construct_covariance(self, cross=None, **extras):
        jitter, locs, amps = self._params
        #round to the nearest index
        locs = locs.astype(int)
        # make sure the flux vector exists and is of proper length
        nw = len(self._sigma)
        if np.all(self._flux == 1):
            flux = np.ones(nw)
        else:
            flux = self._flux
        assert len(flux) > np.max(locs)
        assert len(flux) == nw
        
        
        Sigma = np.zeros([nw, nw])
        Sigma[(np.arange(nw), np.arange(nw))] += self._sigma**2 + (jitter*flux)**2
        Sigma[(locs, locs)] += (amps*flux[locs])**2

        return Sigma
