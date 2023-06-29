
SFH Treatments
==============

Numerous star formation history (SFH) treatments are available in prospector.
Some of these are described below, along with instructions for their use.

SSPs
----
Simple or single stellar populations (SSPs) describe the spectra and properties
of a group of stars (withi initial mass distribution described by the IMF) with
the same age and metallicity.  That is, the SFh is a delta-function in both time
in metallicity.

Use of SSP SFHs requires an instance of
:py:class:`prospect.sources.CSPSpecBasis` to be used as the ``sps`` object. A set
of propsector parameters implementing this treatment is available as the
``"ssp"`` entry of :py:class:`prospect.models.templates.TemplateLibrary`.

Parametric SFH
--------------

So called "parametric" SFHs describe the SFR as a function of time via a
relatively simple function with just a few parameters.  In prospector the
parametric SFH treatment is actually handled by FSPS itself, and so the model
parameters required are the same as those in FSPS (see `documentation
<https://github.com/cconroy20/fsps/blob/master/doc/MANUAL.pdf>`_).

The available parametric SFHs include exponential decay ("tau" models,
:math:`{\rm SFR} \sim e^{-{\rm tage}/{\rm tau}}`), and delayed exponential
("delayed-tau" models, :math:`{\rm SFR} \sim {\rm tage} \, e^{-{\rm tage}/{\rm tau}}`).
To these it is possible to add a burst and/or a truncation, and a constant
component can also be added.  Finally, the SFH descibed in simha14 is also
available. See the `FSPS documentation
<https://github.com/cconroy20/fsps/blob/master/doc/MANUAL.pdf>`_ for details.

It is also possible to model *linear combinations* of these parameteric SFHs.
This is accomplished by making the ``mass`` parameter a vector with the number
of elements corresponding to the number of components.  Other paramaters of the
FSPS stellar population model (e.g. ``age``, ``tau``, and even ``dust2`` or
``dust1``) can be also be made vectors, with vector priors if they are free to
be fit; relevant scalar parameters will be shared by all components.

Use of parametric SFHs requires an instance of
:py:class:`prospect.sources.CSPSpecBasis` to be used as the ``sps`` object. A set
of propsector parameters implementing this treatment (defaulting to a delay-tau
form, ``sfh=4``) is available as the ``"parametric_sfh"`` entry of
:py:class:`prospect.models.templates.TemplateLibrary`.

Binned SFHs
-----------

The binned or "non-parametric" SFHS are a more flexible alternative to the
"parametric" SFHs described above.  Rather than being "non-parametric" they
actually rely on various parameterizations that fundamentally describe a
piece-wise constant SFH, where the SFR is constant within each of a user defined
set of temporal bins.

Use of these piece-wise constant SFHs requires an instance of
:py:class:`prospect.sources.FastStepBasis` to be used as the ``sps`` object.
Fundamentally this class requires two vector parameters to generate a model:

* ``agebins`` an array of shape ``(Nbin, 2)`` describing the lower and upper
  *lookback time* of each bin (in units of log(years))
* ``mass`` an array of shape ``(Nbin,)`` describing the total stellar mass
  **formed** in each bin.  For the ith bin this means
  :math:`{\rm SFR}_i = {\rm mass}_i / (10^{{\rm agebins}_{i, 1}} - 10^{{\rm agebins}_{i, 0}})`

The SFH treatments described below all differ in how they transform from the
sampled SFH parameters to these fundamental binned SFH parameters, and in the
priors placed on those sampled parameters.  The transformations between the
sampling parameters and these fundamental parameters are given by methods within
:py:mod:`prospect.models.transforms`


Continuity SFH
^^^^^^^^^^^^^^
See `leja19 <https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract>`_,
`johnson21 <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_
for more details. A set of propsector parameters implementing this treatment
with 3 bins is available as the ``"continuity_sfh"`` entry of
:py:class:`prospect.models.templates.TemplateLibrary`.

In this parameterization, the SFR of each bin is derived from sampling a vector
of parameters describing the *ratio* of SFRs in adjacent temporal bins.  By
default, a Student-t prior distribution (like a Gaussian but with heavier tails)
is placed on the log of these ratios.  This results in a prior SFH that tends
toward constant SFR, and down-weights drmamtic changes in the SFR between
adjacent bins.  The overall normalization is provided by the ``logmass``
parameter.

In detail, the SFR in each timetime is computed as

.. math::

    {\rm SFR}_i = K \, \prod_{j=1}^{j<i} r_j

where :math:`K` is a normalization constant. These are then converted to masses
by multiplication with the bin widths and renormalization by the total mass.

To change the number of bins see
:py:meth:`prospect.models.templates.adjust_continuity_agebins`.  This method
produces 3 bins with defined edges at recent and very distant lookback times,
and then divides the remaining time in to bins of equal intervals of
:math:`\log(t_{\rm lookback})`

Continuity Flex SFH
^^^^^^^^^^^^^^^^^^^
See `leja19 <https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract>`_
for more details. A set of prospector parameters implementing this treatment is
available as the ``"continuity_flex_sfh"`` entry of
:py:class:`prospect.models.templates.TemplateLibrary`

In this parameterization, the edges of the temporal bins are adjusted such that
for a given set of SFRs an equal amount of mass forms in each bin.  In other
words, the bins all contain the same fraction of the total stellar mass, and the
free parameters are related to the time it takes each succesive quantile of the
mass to form. The widths are derived from the :math:`J` sampled SFR ratios
:math:`r_j = {\rm SFR}_j / {\rm SFR}_{j+1}` as

.. math::

    \Delta t_0 = t_{\rm flex}  / (1 + \sum_{n=1}^{n=J} \prod_{j=1}^{j=n} r_j) \\
    \Delta t_i = \Delta t_0 \, \prod_{j=1}^{j=i} r_j

where :math:`t` is lookback time. Note that the width of the first and last bin
are fixed to the values supplied in the initial ``"agebins"`` parameter, while
:math:`t_{\rm flex}` is the remaining interval of lookback time.


PSB Hybrid SFH
^^^^^^^^^^^^^^
See `suess21 <https://ui.adsabs.harvard.edu/abs/2021arXiv211114878S/abstract>`_
for details.

This parameterization provides a number of fixed width bins at both small and
large lookback times, combined with a number of flexible width bins between
these fixed bins. These are designed to efficiently produce the flexibility
required to model post-starburst SFHs. A set of prospector parameters
implementing this treatment is available as the ``"continuity_psb_sfh"`` entry
of :py:class:`prospect.models.templates.TemplateLibrary`

Dirichlet SFH
^^^^^^^^^^^^^
See `leja17 <https://ui.adsabs.harvard.edu/abs/2017ApJ...837..170L/abstract>`_,
`leja19 <https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract>`_ for
more details.  A set of prospector parameters implementing this treatment is
available as the ``"dirichlet_sfh"`` entry of
:py:class:`prospect.models.templates.TemplateLibrary`

In this parameterization the sampling variables are related to the fraction of
the total stellar mass formed in each bin.  Since these fractions must add up to
1, the parameter space corresponds to a Dirichlet distribution, and for
numerical reasons this is best represented by sampling in a dimensionless vector
variable ``z_fraction`` with a specific prior distribution. Transformations from
these dimensionless variables to SFRs or masses in each bin are provided in
:py:mod:`prospect.models.transforms`.



