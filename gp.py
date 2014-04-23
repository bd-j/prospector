import numpy as np
from scipy.linalg import cho_factor, cho_solve

class GaussianProcess(object):

    def __init__(self, wave, sigma ):
        self.wave = wave
        self.sigma = sigma

    def factor(self, s, a, l):
        self.s = s
        self.a =a
        self.l = l
        Sigma = a**2 * np.exp(-(self.wave[:,None] -self.wave[None,:])**2/(2*l**2))
        Sigma[np.diag_indices_from(Sigma)] += (self.sigma**2 + s**2)
        self.factorized_Sigma  = cho_factor(Sigma, overwrite_a  = True)
        self.log_det = np.sum(2 * np.log(np.diag(self.factorized_Sigma[0])))
        assert np.isfinite(self.log_det)
                
    def lnlike(self, residual):
        return  -0.5* (np.dot(residual, cho_solve(self.factorized_Sigma, residual)) + self.log_det)

    def predict(self, residual):
        Sigma = self.a**2 * np.exp(-(self.wave[:,None] -self.wave[None,:])**2/(2*self.l**2))
        Sigma[np.diag_indices_from(Sigma)] += ( self.s**2)        
        return np.dot(Sigma, cho_solve(self.factorized_Sigma, residual))

