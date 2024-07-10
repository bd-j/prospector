Fitting Spectra
================

There are several extra considerations that come up when fitting spectroscopy.

Wavelength range, resolution, and linespread function
-----------------------------------------------------

Prospector is based on FSPS, which uses stellar spectral libraries with given
resolution. The empirical MILES library has a resolution of ~2.5A FWHM from
3750AA - 7200AA restframe, and much lower (R~200 or so, but not actually well
defined) outside this range. Higher resolution data (after including both
velocity dispersion and instrumental resolution) or spectral points outside this
range cannot yet be fit.

Prospector includes methods for FFT based smoothing of the spectra, assuming a
gaussian LSF (in either wavelength or velocity space). There is also the
possibility of FFT based smoothing for wavelength dependent Gaussian dispersion
(i.e. sigma_lambda = f(lambda) with f possibly a polynomial of lambda). In
practice the smoothed spectra will be a combination of the library resolution
plus whatever FFT smoothing is applied. Hopefully this can be made to match your
actual data resolution, which is a combination of the physical velocity
dispersion and the instrumental resolution. The smoothing is controlled by the
parameters `sigma_smooth`, `smooth_type`, and `fftsmooth`


Instrumental Response & Spectrophotometric Calibration
---------------------

There are various options for dealing with the spectral continuum shape
depending on how well you think your spectra are calibrated and if you also fit
photometry to tie down the continuum shape. You can optimize out a polynomial
"calibration" vector, or simultaneously fit for a polynomial and marginalize
over the polynomial coefficients (this allows you to place priors on the
accuracy of the spectrophotometric calibration). Or you can just take the
spectrum as perfectly calibrated.

Particular treatments can be implemented using different mixin classes, e.g.

.. code-block:: python

        from prospect.observation import Spectrum, PolyOptCal
        class PolySpect(PolyOptCal, Spectrum):
            pass
        spec = PolySpect(wavelength=np.linspace(3000, 5000, N),
                         flux=np.zeros(N),
                         uncertainty=np.ones(N),
                         polynomial_order=5)


Nebular emission
----------------

While prospector/FSPS include self-consistent nebular emission, the treatment is
probably not flexible enough at the moment to fit high S/N, high resolution data
including nebular emission (e.g. due to deviations of line ratios from Cloudy
predictions or to complicated gas kinematics that are different than stellar
kinematics). Thus fitting nebular lines should take adavantage of the nebular
line amplitude optimization/marginalization capabilities. For very low
resolution data this is less of an issue.