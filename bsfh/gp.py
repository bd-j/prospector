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
        self.asq = None
        self.lsq = None
        self.s = None
        self.wave = wave
        self.sigma = sigma
        
    def kernel_to_params(self, kernel):
        """Kernel is a vector consisting of log(s, a**2, l**2)
        """
        s, asquared, lsquared = exp(kernel).tolist()
        return s, asquared, lsquared
        
    def compute(self, wave=None, sigma=None, check_finite=False,force=False):
        """
        :param wave:
            independent variable
            
        :param sigma:
            .
            
        """
        if wave is None:
            wave = self.wave
        if sigma is None:
            sigma = self.sigma
        s, asq, lsq = self.kernel_to_params(self.kernel)
        kernel_same = (s == self.s) & (asq == self.asq) & (lsq == self.lsq)
        data_same = (np.all(wave == self.wave) &
                     np.all(sigma == self.sigma))

        if kernel_same and data_same and (not force):
            return
        
        else:
            self.s = s
            self.asq = asq
            self.lsq = lsq
            self.wave = wave
            self.sigma = sigma
            
            Sigma = self.asq * np.exp(-(self.wave[:,None] - self.wave[None,:])**2/(2*self.lsq))
            Sigma[np.diag_indices_from(Sigma)] += (self.sigma**2 + self.s**2)
            self.factorized_Sigma  = cho_factor(Sigma, overwrite_a=True,
                                                check_finite=check_finite)
            self.log_det = 2 * np.sum( np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)
                            
    def lnlikelihood(self, residual, check_finite=False):
        """
        Compute the ln of the likelihood.
        
        :param residual: ndarray, shape (nwave,)
            Vector of residuals (y_data - mean_model).
        """
        s, asq, lsq = self.kernel_to_params(self.kernel)
        kernel_same = (s == self.s) & (asq == self.asq) & (lsq == self.lsq)
        if not kernel_same:
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
        
        if wave is None:
            wave = self.wave
        Sigma = self.a**2 * np.exp(-(wave[:,None] -self.wave[None,:])**2/(2*self.l**2))
        Sigma[np.diag_indices_from(Sigma)] += ( self.s**2)        
        return np.dot(Sigma, cho_solve(self.factorized_Sigma, residual))

    def predict_var(self, wave=None):
        """
       Give the GP prediction variance at each wavelength.

        :param wave: default None
            Wavelengths at which variance estimates are desired.
            Defaults to the input wavelengths - the variance is zero
            in this case.
        """
        
        if wave is None:
            inwave = self.wave
        else:
            inwave = wave
        Sigma = self.a**2 * np.exp(-(inwave[:,None] -self.wave[None,:])**2/(2*self.l**2))
        Sigma[np.diag_indices_from(Sigma)] += ( self.s**2)
        if wave is None:
            Sigma_star = Sigma
        else:
            Sigma_star = self.a**2 * np.exp(-(inwave[:,None] - inwave[None,:])**2/(2*self.l**2))
            Sigma_star[np.diag_indices_from(Sigma_star)] += ( self.s**2)
       
        return Sigma_star - np.dot(Sigma, cho_solve(self.factorized_Sigma, Sigma))
