
Prospector-Beta Priors
======================

This model is intended for fitting galaxy photometry where the redshift is
unknown. The priors encode empirical constraints of redshifts, masses, and star
formation histories in the galaxy population.

.. note::
    The implementation of these priors was recently updated to be fully dynamic.
    Unlike previous versions which relied on static tabulated files for the redshift
    prior, the current version calculates the galaxy number density and differential
    comoving volume on-the-fly. This ensures consistency with the chosen cosmology
    and allows for redshift-dependent mass limits (e.g., for flux-limited surveys).

Please cite Wang et al. (`2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..58W/abstract>`__)
and the relevant papers if any of the priors are used.

Motivation
----------
Standard photometric redshift codes often assume uniform priors on physical parameters.
However, Wang et al. (`2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..58W/abstract>`__)
demonstrated that uniform priors can lead to an age-mass-redshift degeneracy -
an overestimation in age leads to a higher mass-to-light ratio, requiring an
overestimated mass to match the luminosity, while redshift often remains unconstrained.

The Prospector-:math:`\beta` priors are designed to break this degeneracy by
downweighting physically unlikely solutions (e.g., massive, quiescent galaxies
in the very early universe). In tests on mock JWST observations, these priors reduced
bias in mass and age estimates by ~0.2-0.4 dex compared to uniform priors.

Usage
-----

A set of Prospector parameters implementing the full set of priors is available
as the ``"beta"`` entry of
:py:class:`prospect.models.templates.TemplateLibrary`:

.. code-block:: python

    from prospect.models.templates import TemplateLibrary
    model_params = TemplateLibrary["beta"] # Uses the NzSFH priors

Additionally, we provide different combinations of the priors for flexibility, depending
on whether you want your model to assume the stellar mass function (SMF) prior,
mass-metallicity relation (MZR) prior, dynamic redshift prior :math:`p(z)` from the galaxy
number density, or the cosmic star formation rate density (SFRD) prior (as described below):

* ``PhiMet``         : SMF + MZR, uniform prior on :math:`z`
* ``ZredMassMet``    : Dynamic :math:`p(z)` + SMF + MZR
* ``DymSFH``         : MZR + SFH, uniform priors on :math:`\log_{10}(M_*/M_\odot)` and :math:`z`
* ``DymSFHfixZred``  : same as ``DymSFH``, but with fixed :math:`z`
* ``PhiSFH``         : SMF + MZR + SFRD, uniform prior on :math:`z`
* ``PhiSFHfixZred``  : same as ``PhiSFH``, but with fixed :math:`z`
* ``NzSFH``          : Dynamic :math:`p(z)` + SMF + MZR + SFRD  (the full Prospector-:math:`\beta` prior).

These priors can be accessed from :py:mod:`prospect.models.priors_beta`.

-----

Prior Descriptions
------------------

Stellar Mass Function
~~~~~~~~~~~~~~~~~~~~~
This prior constrains the stellar mass distribution at a given redshift. Since low-mass
galaxies are far more numerous than high-mass galaxies in the universe, this prior helps
avoid spurious high-mass, high-redshift solutions.

This prior takes one of two different forms, depending on the ``"const_phi"`` parameter:

1. ``"const_phi = True"``
   Uses the continuity model from Leja et al. (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract>`__)
   for :math:`0.2 \leq z \leq 3.0`. Outside this range, the nearest neighbor (:math:`z=0.2` or :math:`z=3.0`) is used.

2. ``"const_phi = False"``
   Transitions to the UV-selected mass functions from Tacchella et al. (`2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T/abstract>`__)
   for :math:`z > 4`, with a smooth interpolation between :math:`z=3` and :math:`z=4`.


Galaxy Number Density (Dynamic Redshift Prior)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This prior informs the model about the survey volume being probed. It represents the
probability of finding a galaxy at redshift :math:`z` given the survey's volume and
mass completeness.

.. math::

    p(z) \propto N(M_* > M_{\mathrm{min}}, z) \times \frac{dV_c}{dz}

where :math:`\frac{dV_c}{dz}` is the differential comoving volume and :math:`N` is
the number density of galaxies above the mass limit :math:`M_{\mathrm{min}}`. This
lower mass limit is set using the ``"mass_mini"`` parameter.

**Advanced Usage: Flux-Limited Surveys**

In flux-limited surveys, the mass completeness limit :math:`M_{\mathrm{min}}(z)` becomes
a function, with the limit generally increasing with redshift. You can account for this
by passing a **function** to the ``"mass_mini"`` parameter. This function must accept a
redshift and return the log stellar mass limit.

.. code-block:: python

    import numpy as np
    from prospect.models import priors_beta

    # Define a function approximating the mass limit of your survey
    # e.g., log10(M_lim(z)) ~ 8.0 + 1.0 * z
    def my_mass_limit(z):
        return 8.0 + 1.0 * z

    # Pass this function to the prior
    prior = priors_beta.NzSFH(
        zred_mini=0.1,
        zred_maxi=10.0,
        mass_mini=my_mass_limit, # Pass the function here
        mass_maxi=13.0,
        # ... other args ...
    )


Dynamic Star-formation History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standard non-parametric SFH priors (like ``"continuity_sfh"``) often assume a flat expectation
for the SFR in each time bin. The Dynamic SFH prior modifies this to match physical
expectations:

1. **Cosmic SFRD:** The expectation value of the SFR in each bin is matched to the cosmic
   Star Formation Rate Density (Behroozi et al. `2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B/abstract>`__).
2. **Downsizing:** The prior introduces a mass dependence. High-mass galaxies are expected to form earlier
   (older ages), while low-mass galaxies form later. This is implemented by shifting the start of the age
   bins as a function of galaxy mass.


Stellar Mass-Stellar Metallicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This prior enforces the mass-metallicity relation measured from SDSS by Gallazzi et al.
(`2005 <https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G/abstract>`__). Following
Leja et al. (`2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract>`__),
the width of the observational relationship is **inflated by a factor of 2** to
conservatively account for systematic uncertainties in SED modeling and potential
redshift evolution.
