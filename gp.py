import numpy as np
from scipy.linalg import cho_factor, cho_solve

class GaussianProcess(object):

    def __init__(self, wave, sigma ):
        """
        Initialize the relevant parameters for the gaussian process.

        :param wave:
            The wavelength scale of the points for which you want
            estimates.
           
        :param sigma:
            The uncertainty estimate at each wavelength point.
        """
        self.wave = wave
        self.sigma = sigma
        self.s = None
        self.a = None
        self.l = None
        
    def factor(self, s, a, l, check_finite=True,force=False):
        """
        :param s:
            Jitter (diagonal) term
            
        :param a:
            Amplitude of covariance gaussian (in units of flux).
            
        :param l:
            Length scale of gaussian covariance kernel, in units of
            wavelength.
            
        """
        if (s == self.s) & (a == self.a) & (l == self.l) & (not force):
            return
        else:
            self.s = s
            self.a = a
            self.l = l
            Sigma = a**2 * np.exp(-(self.wave[:,None] - self.wave[None,:])**2/(2*l**2))
            Sigma[np.diag_indices_from(Sigma)] += (self.sigma**2 + s**2)
            self.factorized_Sigma  = cho_factor(Sigma, overwrite_a=True, check_finite=check_finite)
            self.log_det = np.sum(2 * np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)
            
            fstring = 's={0}, a={1}, l={2}'
            print(fstring.format(self.s, self.a, self.l))
            print('<sigma**2>={0}'.format( (self.sigma**2).mean() ))
            
            #print(self.log_det)
                
    def lnlike(self, residual, check_finite=True):
        """
        Compute the ln of the likelihood.
        
        :param residual: ndarray, shape (nwave,)
            Vector of residuals (y_data - mean_model).
        """
        print('<r**2>={0}'.format((residual**2).mean()))

        first_term = np.dot(residual,
                              cho_solve(self.factorized_Sigma, residual, check_finite = True))
        lnL=  -0.5* (first_term
                              + self.log_det)
        print('lnlike(r) = {0}'.format(lnL))
        print('log_det={0}'.format(self.log_det))
        print('dot(r, solve(C,r))={0}'.format(first_term))
        print('N_p={0}'.format(len(residual)))
        
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
            in theis case.
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
