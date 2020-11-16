Frequently Asked Questions
==========================

How do I add filter transmission curves?
----------------------------------------
Many projects use particular filter systems that are not general or widely used.
It is easy to add a set of custom filter curves. Filter projections are handled
by the `sedpy <https://github.com/bd-j/sedpy>`_ code. See the FAQ there
<https:github.com/bd-j/sedpy/blob/main/docs/faq.rst> for detailed instructions
on adding filter cirves.


What units?
-----------
Prospector natively uses *maggies* for both spectra and photometry, and it is
easiest if you supply data in these units. maggies are a flux density (:math:`f_{\nu}`) unit
defined as Janskys/3631. Wavelengths are *vacuum* Angstroms by default. By
default, masses are in solar masses of stars *formed*, i.e. the integral of the
SFH. Note that this is different than surviving solar masses (due to stellar
mass loss).


How long will it take to fit my data?
-------------------------------------
That depends.
Here are some typical timings for each likelihood call in various situations (macbook pro)

   + 10 band photometry, no nebular emission, no dust emission: 0.004s
   + photometry, including nebular emission: 0.04s
   + spectroscopy including FFT smoothing: 0.05s


Note that the initial likelihood calls will be (much) longer than this.  Under
the hood, the first time it is called python-FSPS computes and caches many
quantities that are reused in subsequent calls.

Many likelihood evaluations may be required for the sampling to converge. This
depends on the kind of model you are fitting and its dimensionality, the
sampling algorithm being used, and the kind of data that you have.  Hours or
even days per fit is not uncommon for more complex models.


Can I fit my spectrum too?
--------------------------
There are several extra considerations that come up when fitting spectroscopy

   1) Wavelength range and resolution.
      Prospector is based on FSPS, which uses the MILES spectral library. These
      have a resolution of ~2.5A FWHM from 3750AA - 7200AA restframe, and much
      lower (R~200 or so, but not actually well defined) outside this range.
      Higher resolution data (after including both velocity dispersion and
      instrumental resolution) or spectral points outside this range cannot yet
      be fit.

   2) Relatedly, line spread function.
      Prospector includes methods for FFT based smoothing of the spectra,
      assuming a gaussian LSF (in either wavelength or velocity space). There is
      also the possibility of FFT based smoothing for wavelength dependent
      gaussian dispersion (i.e. sigma_lambda = f(lambda) with f possibly a
      polynomial of lambda). In practice the smoothed spectra will be a
      combination of the library resolution plus whatever FFT smoothing is
      applied. Hopefully this can be made to match your actual data resolution,
      which is a combination of the physical velocity dispersion and the
      instrumental resolution. The smoothing is controlled by the parameters
      `sigma_smooth`, `smooth_type`, and `fftsmooth`

   3) Nebular emission.
      While prospector/FSPS include self-consistent nebular emission, the
      treatment is probably not flexible enough at the moment to fit high S/N,
      high resolution data including nebular emission (e.g. due to deviations of
      line ratios from Cloudy predictions or to complicated gas kinematics that
      are different than stellar kinematics). Thus fitting nebular lines should
      take adavantage of the nebular line amplitude optimization/marginalization
      capabilities. For very low resolution data this is less of an issue.

   4) Spectrophotometric calibration.
      There are various options for dealing with the spectral continuum shape
      depending on how well you think your spectra are calibrated and if you
      also fit photometry to tie down the continuum shape. You can optimize out
      a polynomial "calibration" vector, or simultaneously fit for a polynomial
      and marginalize over the polynomial coefficients (this allows you to place
      priors on the accuracy of the spectrophotometric calibration). Or you can
      just take the spectrum as perfectly calibrated.


What SFH parameters should I use?
---------------------------------
That depends on the scientific question you are trying to answer,
and to some extent on the data that you have.


How do I use the non-parametric SFHs?
-------------------------------------
|Codename| is packaged with four families of non-parametric star formation
histories.  The simplest model fits for the (log or linear) mass formed in fixed
time bins.  This is the most flexible model but typically results in unphysical
priors on age and specific star formation rates.  Another model is the Dirichlet
parameterization introduced in
`leja17 <https://ui.adsabs.harvard.edu/abs/2017ApJ...837..170L/abstract>`_,
in which the fractional specific star formation rate for fixed each time bin
follows a Dirichlet distribution. This model is moderately flexible and produces
reasonable age and sSFR priors, but it still allows unphysical SFHs with sharp
quenching and rejuvenation events.  A third model, the continuity prior, trades
some flexibility for accuracy by explicitly weighting against these (typically
considered unphysical) sharply quenching and rejuvenating SFHs. This is done by
placing a prior on the ratio of SFRs in adjacent time bins to ensure a smooth
evolution of SFR(t). Finally, the flexible continuity prior retains this
smoothness weighting but instead of fitting for the mass forms in fixed bins,
fits for the size of time bins in which a fixed amount of mass is formed.  This
prior removes the discretization effect of the time bins in exchange for
imposing a minimum mass resolution in the recovered SFH parameters.  The
performance of these different nonparametric models is compared and contrasted
in detail in
`leja19 <https://ui.adsabs.harvard.edu/abs/2019ApJ...873...44C/abstract>`_.

In order to use these models, select the appropriate *parameter set template* to
use in the ``model_params`` dictionary.  You will also need to make sure to use
the appropriate *source* object, :py:class:`prospect.sources.FastStepBasis`


What bins should I use for the non-parametric SFH?
-------------------------------------------------
Deciding on the "optimal" number of bins to use in such non-parametric SFHs is a
difficult question.  The pioneering work of
`ocvirk06 <https://ui.adsabs.harvard.edu/abs/2006MNRAS.365...46O/abstract>`_
suggests approximately 10 independent components can be recovered from extremely
high S/N R=10000 spectra (and perfect models). The fundamental problem is that
the spectra of single age populations change slowly with age (or metallicity),
so the contributions of each SSP to a composite spectrum are very highly
degenerate and some degree of regularization or prior information is required.
However, the ideal regularization depends on the (*a priori* unknown) SFH of the
galaxy.  For example, for a narrow burst one would want many narrow bins near
the burst and wide bins away from it. Reducing the number of bins effectively
amounts to collapsing the prior for the ratio of the SFR in two sub-bins to a
delta-function at 1.  Using too few bins can result in biases in the same way as
the strong priors imposed by parametric models. Tests in
`leja19 <<https://ui.adsabs.harvard.edu/abs/2019ApJ...873...44C/abstract>>`_
suggest that ~5 bins are adequate to model covariances in basic parameters from
photometry, but more bins are better to explore detailed constraints on SFHs.


How do I fit for redshift as well as other parameters?
------------------------------------------------------


So should I use `emcee`, `nestle`, or `dynesty` for posterior sampling?
-----------------------------------------------------------------------
We recommend using the `dynesty` nested sampling package.

In addition to the standard sampling phase which terminates based on the quality
of the estimation of the Bayesian evidence, `dyensty` includes a subsequent
dynamic sampling phase which, as implemented in |Codename|, instead terminates
based the quality of the posterior estimation. This permits the user to specify
stopping criteria based directly on the quality of the posterior sampling with
the ```nested_posterior_thresh``` keyword, providing direct control over the
trade-off between posterior quality and computational time. A value of 0.02 for
this keyword specifies high-quality posteriors, whereas a value of 0.05 will
produce reasonable but approximate posteriors. Additionally, `dyensty` sampling
can be parallelized in |Codename|: this produces faster convergence time at the
cost of lower computational efficiency (i.e., fewer model evaluations per unit
computational time). It is best suited for fast evaluation of small samples of
objects, whereas single-core fits produce more computationally efficient fits to
large samples of objects.

What settings should I use for `dynesty`?
--------------------------------------
The default \dynesty{} settings in |Codename| are optimized for a
low-dimensional ($N=4-7$) model. Higher-dimensional models with more complex
likelihood spaces will likely require more advanced `dynesty` settings to
ensure efficient and timely convergence. This often entails increasing the
number of live points, changing to more robust sampling methodology (e.g., from
uniform to a random walk), setting a maximum number of function calls, or
altering the target evidence and posterior thresholds. More details can be found
in `speagle20 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract>`_
and the `dynesty online documentation <https://dynesty.readthedocs.io/en/latest/faq.html>`_.
The list of options and their default values can be seen with

.. code-block:: python

        from prospect.utils import prospect_args
        prospect_args.show_default_args()


The chains did not converge when using `dynesty`, why?
------------------------------------------------------
It is likely that they did converge; note that the convergence for MC sampling
of a posterior PDF is not defined by the samples all tending toward the a single
value, but as the *distribution* of samples remaining stable.  The samples for a
poorly constrained parameter will remain widely dispersed, even if the MC
sampling has converged to the correct *distribution*


How do I use `emcee` in |Codename|?
-------------------------------------
For each parameter, an initial value must be given.  The ensemble of walkers is
initialized around this value, with a Gaussian spread that can be specified
separately for each parameter.  Each walker position is evolved at each
iteration using parameter proposals derived from an ensemble of the other
walkers. In order to speed up initial movement of the cloud of walkers to the
region of parameter space containing most of the probability mass, multiple user
defined rounds of burn-in may be performed. After each round the walker
distribution in parameter space is re-initialized to a multivariate Gaussian
derived from the best 50% of the walkers (where best is defined in terms of
posterior probability at the last iteration).  The iterations in these burn-in
rounds are discarded before a final production run. It is important to ensure
that the chain of walkers has converged to a stable *distribution* of
parameter values. Diagnosing convergence is fraught; a number of indicators have
been proposed `sharma17 <https://ui.adsabs.harvard.edu/abs/2017ARA%26A..55..213S/abstract>`_
including the auto-correlation time of the chain
`goodman10 <https://ui.adsabs.harvard.edu/abs/2010CAMCS...5...65G/abstract>`_.
Comparing the results of separate chains can also provide a sanity check.


When should I use optimization?
-------------------------------
Optimization can be performed before ensemble MCMC sampling, to decrease the
burn-in time of the MCMC algorithm. |Codename| currently supports
Levenburg-Marquardt least-squares optimization and Powell's method, as
implemented in `SciPy <https://www.scipy.org>`_. It is possible to start
optimizations from a number of different parameter values, drawn from the prior
parameter distribution, in order to mitigate the problems posed by local maxima.


How do I know if Prospector is "working"?
---------------------------------------


What do I do with the chain?  What values should I report?
----------------------------------------------------------
This is a general question for MC sampling techniques.
Please see X, Y, Z for advice.


Why isn't the posterior PDF centered on the highest posterior probability sample?
---------------------------------------------------------------------


How do I interpret the `lnprobability` or `lnp` values? Why do I get `lnp > 0`?
-------------------------------------------------------------------------------


How do I plot the best fit SED?  How do I plot uncertainties on that?
---------------------------------------------------------------------
Note that the highest probability sample is *not* the same as the maximum a
posteriori (MAP) solution.  The MAP solution inhabits a vanishingly small region
of the prior parameter space; it is exceedingly unlikely that the MCMC sampler
would visit exactly that location.  Furthermore, when large degeneracies are
present, the maximum a posteriori parameters may be only very slightly more
likely than many solutions with very different parameters.


How do I get the wavelength array for plotting spectra and/or photometry when fitting only photometry?
------------------------------------------------------------------------------------------------------


Should I fit spectra in the restframe or the observed frame?
------------------------------------------------------------
You can do either if you are fitting only spectra. If fitting in the restframe
then the distance has to be specified explicitly, otherwise it is inferred from
the redshift.

If you are fitting photometry and spectroscopy then you should be fitting the
observed frame spectra.


What do I do about upper limits?
--------------------------------


What priors should I use?
-------------------------
That depends on the scientific question and the objects under consideration.
In general we recommend using informative priors (e.g. narrow ``Normal``
distributions) for parameters that you think might matter at all.


What happens if a parameter is not well constrained?  When should I fix parameters?
-----------------------------------------------------------------------------------
If some parameter is completely unconstrained you will get back the prior. There
are also (often) cases where you are "prior-dominated", i.e. the posterior is
mostly set by the prior but with a small perturbation due to small amounts of
information supplied by the data. You can compare the posterior to the prior,
e.g. using the Kullback-Liebler divergence between the two distributions, to see
if you have learned anything about that parameter. Or just overplot the prior on
the marginalized pPDFs

To be fully righteous you should only fix parameters if

  - you are very sure of their values;
  - or if you don't think changing the parameter will have a noticeable effect on the model;
  - or if a parameter is perfectly degenerate (in the space of the data) with another parameter.

In practice parameters that have only a small effect but take a great deal of
time to vary are often fixed.



.. |Codename| replace:: Prospector
