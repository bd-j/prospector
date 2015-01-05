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
            self.kernel = np.array([0.0, 0.0, 0.0])
        else:
            self.kernel = kernel
        #_params stores values of kernel parameter used to construct
        #and compute the factorized covariance matrix
        self._params = None
        self.wave = wave
        self.sigma = sigma
        
    def kernel_to_params(self, kernel):
        """Kernel is a vector consisting of log(s, a**2, l**2)
        """
        s, asquared, lsquared = np.exp(kernel).tolist()
        return s, asquared, lsquared

    @property
    def kernel_same(self):
        s, asq, lsq = self.kernel_to_params(self.kernel)
        kernel_same = (s == self.s) & (asq == self.asq) & (lsq == self.lsq)
        return kernel_same

    def construct_covariance(self, inwave=None, cross=False):

        s, asq, lsq = self.params
        
        if inwave is None:
            X_star = X = self.wave
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

    def compute(self, wave=None, sigma=None, check_finite=False, force=False):
        """
        :param wave: optional
            independent variable
            
        :param sigma:
            .
            
        """
        if wave is None:
            wave = self.wave
        if sigma is None:
            sigma = self.sigma
        data_same = (np.all(wave == self.wave) &
                     np.all(sigma == self.sigma))
        params = self.kernel_to_params(self.kernel)
        kernel_same = np.array_equal(params, self._params)

        if kernel_same and data_same and (not force):
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
        Compute the ln of the likelihood, using the current factorized sigma
        
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
        For a given residual vector, give the GP mean prediction at each wavelength.

        :param residual:
            Vector of residuals (y_data - mean_model).
            
        :param wave: default None
            Wavelengths at which variance estimates are desired.
            Defaults to the input wavelengths.
        """
        
        
        Sigma_cross = self.construct_covariance(wave=wave, cross=True)
        Sigma_star = self.construct_covariance(wave=wave, cross=False)
        
        mu = np.dot(Sigma, cho_solve(self.factorized_Sigma, residual))
        cov = Sigma_star - np.dot(Sigma_cross, cho_solve(self.factorized_Sigma,
                                                         -Sigma_cross.T))
        return mu, cov
                              
