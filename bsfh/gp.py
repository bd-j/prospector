import numpy as np
from scipy.linalg import cho_factor, cho_solve

class GaussianProcess(object):

    def __init__(self, wave=None, sigma=None, kernel=None ):
        """
        Initialize the relevant parameters for the gaussian process.

        :param wave:
            The wavelength scale of the points for which you want
            estimates.
           
        :param sigma:
            The uncertainty estimate at each wavelength point.
        """
        if kernel is None:
            npar = self.kernel_properties[0]
            self.kernel = np.zeros(npar)
        else:
            self.kernel = kernel
        #_params stores the values of kernel parameters used to
        #construct and compute the factorized covariance matrix that
        #is stored in factorized_Sigma
        self._params = None
        self.wave = wave
        self.sigma = sigma

    def reset(self):
        self.factorized_Sigma = None
        self.wave = None
        self.sigma = None
        self._params = None
        self.kernel = None
        
    @property
    def kernel_same(self):
        params = self.kernel_to_params(self.kernel)
        ksame = np.array_equal(params, self._params)
        return ksame, params
              
    def compute(self, wave=None, sigma=None, check_finite=False, force=False):
        """Construct the covariance matrix, factorize it, and store
        the factorized matrix.  The factorization is only performed if
        the kernel parameters have chenged or the observational data
        (wave and sigma) have changed.
        
        :param wave: optional
            independent variable.
            
        :param sigma:
            uncertainties on the dependent variable at the locations
            of the independent variable
            
        :param force: optional
            If true, force a recomputation even if the kernel and the
            data are the same as for the stored factorization.
        """
        if wave is None:
            wave = self.wave
        if sigma is None:
            sigma = self.sigma
        data_same = (np.all(wave == self.wave) &
                     np.all(sigma == self.sigma))
        params = self.kernel_to_params(self.kernel)
        ksame, params = self.kernel_same()

        if ksame and data_same and (not force):
            return
        
        else:
            self._params = params
            self.wave = wave
            self.sigma = sigma
            
            Sigma = self.construct_kernel()
            self.factorized_Sigma  = cho_factor(Sigma, overwrite_a=True,
                                                check_finite=check_finite)
            self.log_det = 2 * np.sum( np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)
                            
    def lnlikelihood(self, residual, check_finite=False):
        """
        Compute the ln of the likelihood, using the current factorized
        covariance matrix.
        
        :param residual: ndarray, shape (nwave,)
            Vector of residuals (y_data - mean_model).
        """
        assert ( len(residual) == len(self.wave))
        self.compute()
        first_term = np.dot(residual,
                            cho_solve(self.factorized_Sigma,
                                      residual, check_finite = check_finite))
        lnL=  -0.5* (first_term + self.log_det)
        
        return lnL
              
    def predict(self, residual, wave=None):
        """
        For a given residual vector, give the GP mean prediction at
        each wavelength and the covariance matrix.

        :param residual:
            Vector of residuals (y_data - mean_model).
            
        :param wave: default None
            Wavelengths at which mean and variance estimates are desired.
            Defaults to the input wavelengths.
        """
        
        
        Sigma_cross = self.construct_covariance(wave=wave, cross=True)
        Sigma_star = self.construct_covariance(wave=wave, cross=False)
        
        mu = np.dot(Sigma, cho_solve(self.factorized_Sigma, residual))
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

    def construct_covariance(self, inwave=None, cross=False):
        """Construct an exponential squared covariance matrix
        """
        s, asq, lsq = self._params
        
        if inwave is None:
            Sigma = asq * np.exp(-(self.wave[:,None] - self.wave[None,:])**2/(2*lsq))
            Sigma[np.diag_indices_from(Sigma)] += (self.sigma**2 + s**2)
            return Sigma
        elif cross:
            Sigma = asq * np.exp(-(inwave[:,None] - self.wave[None,:])**2/(2*lsq))
            return Sigma
        else:
            Sigma = asq * np.exp(-(inwave[:,None] - inwave[None,:])**2/(2*lsq))
            Sigma[np.diag_indices_from(Sigma)] += s**2
            return Sigma
        
class Outlier(GaussianProcess):

    def kernel_properties(self):
        return [2]
    
    def kernel_to_params(self, kernel):
        """Kernel is a set of (diagonal) locations and amplitudes
        """
        assert ((len(kernel) % 2 == 0))
        nout = len(kernel) / 2
        locs, amps = kernel[np.arange(nout) * 2], kernel[np.arange(nout) * 2 + 1]
        return locs, amps

    def construct_covariance(self, inwave=None, cross=None):
        locs, amps = self._params
        locs = locs.astype(int)
        if inwave is None:
            nw = len(self.wave)
            Sigma = np.zeros([nw, nw])
            #diag = np.diag_indices_from(Sigma)
            Sigma[(np.arange(nw), np.arange(nw))] += self.sigma**2
            Sigma[(locs, locs)] += amps**2
        else:
            raise ValueError
        
        return Sigma
