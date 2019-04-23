Frequently Asked Questions
============

How do I add filter transmission curves?
--------

Many projects use particular filter systems that are not general or widely used.
It is easy to add a set of custom filter curves.
Filter projections are handled by the `sedpy <https:github.com/bd-j/sedpy>`_ code.
See the FAQ there <https:github.com/bd-j/sedpy/blob/master/docs/faq.rst> for detailed instructions on adding filter cirves.

What units?
---------
Prospector natively uses *maggies* for both spectra and photometry,
and it is easiest if you supply data in these units.
maggies are a :math:`f_{\nu}` unit defined as Janskys/3631.
Wavelengths are *vacuum* Angstroms by default.
By default, masses are in solar masses of stars *formed*, i.e. the integral of the SFH.
Note that this is different than surviving solar masses (due to stellar mass loss).

Can I fit my spectrum too?
------------
There are several extra considerations that come up when fitting spectroscopy

   1) wavelength range and resolution - Prospector is based on FSPS, which uses the MILES spectral library.
      These have a resolution of ~2.5A FWHM from 3750AA - 7200AA restframe, and much lower (R~200 or so, but not actually well defined) outside this range.
      Higher resolution data (after including both velocity dispersion and instrumental resolution) or spectral points outside this range cannot yet be fit.

   2) Relatedly, line spread function.
      Prospector includes methods for FFT based smoothing of the spectra, assuming a gaussian LSF (in either wavelength or velocity space).
      There is also the possibility of FFT based smoothing for wavelength dependent gaussian dispersion (i.e. sigma_lambda = f(lambda) with f possibly a polynomial of lambda).
      In practice the smoothed spectra will be a combination of the library resolution plus whatever FFT smoothing is applied.
      Hopefully this can be made to match your actual data resolution, which is a combination of the physical velocity dispersion and the instrumental resolution.
      The smoothing is controlled by the parameters `sigma_smooth`, `smooth_type`, and `fftsmooth`

   3) Nebular emission.
      While prospector/FSPS include self-consistent nebular emission,
      the treatment is probably not flexible enough at the moment to fit high S/N, high resolution data including nebular emission
      (e.g. due to deviations of line ratios from Cloudy predictions or to complicated gas kinematics that are different than stellar kinematics).
      For very low resolution data this is less of an issue.

   4) spectrophotometric calibration.
      There are various options for dealing with the spectral continuum shape depending on
      how well you think your spectra are calibrated and if you also fit photometry to tie down the continuum shape.
      You can optimize out a polynomial "calibration" vector,
      or simultaneously fit for a polynomial and marginalize over the polynomial coefficients
      (this allows you to place priors on the accuracy of the spectrophotometric calibration).
      Or you can just take the spectrum as perfectly calibrated.


How long will it take to fit my data?
---------
That depends.
Here are some typical timings for each likelihood call in various situations (macbook pro)
   * 10 band photometry, no nebular emission, no dust emission: 0.004s
   * photometry, including nebular emission: 0.04s
   * spectroscopy including FFT smoothing: 0.05s



So should I use `emcee`, `nestle`, or `dynesty` for posterior sampling?
--------

How do I know if Prospector is working?
--------

What do I do with the chain?  What values should I report?
--------
This is a general question for MC sampling techniques.
Please see X, Y, Z for advice.

Why isn't the posterior PDF centered on the maximum likelihood value?
--------

How do I interpret the `lnprobability` or `lnp` values? Why do I get `lnp > 0`?
-------

How do I plot the best fit?
-------

How do I get the wavelength array for plotting spectra and/or photometry when fitting only photometry?
--------

Should I fit spectra in the restframe or the observed frame?
-------
You can do either if you are fitting only spectra.
If fitting in the restframe then the distance has to be specified explicitly,
otherwise it is inferred from the redshift.

If you are fitting photometry and spectroscopy then you should be fitting the observed frame spectra.

What do I do about upper limits?
--------

What SFH parameters should I use?
---------
That depends on the scientific question you are trying to answer,
and to some extent on the data that you have.

What priors should I use?
---------


What happens if a parameter is not well constrained?  When should I fix parameters?
-------
If some parameter is completely unconstrained you will get back the prior.
There are also (often) cases where you are “prior-dominated”,
i.e. the posterior is mostly set by the prior but with a small perturbation due to small amounts of information supplied by the data.
You can compare the posterior to the prior, e.g. using the Kullback-Liebler divergence between the two distributions, to see if you have learned anything about that parameter.
Or just overplot the prior on the marginalized pPDFs

To be fully righteous you should only fix parameters if
 you are very sure of their values;
 or if you don't think changing the parameter will have a noticeable effect on the model;
 or if a parameter is perfectly degenerate (in the space of the data) with another parameter.
In practice parameters that have only a small effect but take a great deal of time to vary are often fixed.
