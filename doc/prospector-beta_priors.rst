
Prospector-beta Priors
==============

This model is intended for fitting galaxy photometry where the redshift is
unknown. The priors encode empirical constraints of redshifts, masses, and star
formation histories in the galaxy population.

N.B.: Please make sure to update to the Prospector-beta version post commit
`09a83f2
<https://github.com/bd-j/prospector/commit/09a83f28cae3bcc0f0397b3a0b8d85aa4f96bf12>`_,
merged on May 19, 2023. This is a major update to the SFH(M, z) prior, ensuring
the expectation values of SFRs are well-behaved over the full prior space.

A set of prospector parameters implementing the full set of priors is available
as the ``"beta"`` entry of
:py:class:`prospect.models.templates.TemplateLibrary`.

Additionally we provide different combinations of the priors for flexibility, which includes the following:

* ``PhiMet``         : mass function + mass-met
* ``ZredMassMet``    : number density + mass function + mass-met
* ``DymSFH``         : mass-met + SFH(M, z)
* ``DymSFHfixZred``  : same as above, but keeping zred fixed to a user-specified value, 'zred', during fitting
* ``PhiSFH``         : mass function + mass-met + SFH(M, z)
* ``PhiSFHfixZred``  : same as above, but keeping zred fixed to a user-specified value, 'zred', during fitting
* ``NzSFH``          : number density + mass function + mass-met + SFH(M, z); this is the full set of Prospector-beta priors.

We describe each of the priors briefly below. Please cite `wang23
<https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..58W/abstract>`_ and the
relevant papers if any of the priors are used.


Stellar Mass Function
---------------------

Two options are available, the choice of which depends on the given scientific question.
The relevant data files are ``prior_data/pdf_of_z_l20.txt`` & ``prior_data/pdf_of_z_l20t18.txt``.
These mass functions can also be replaced by supplying new data files to ``prior_data/``

1. ``"const_phi = True"``

The mass functions between 0.2 ≤ z ≤ 3.0 are taken from `leja20
<https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract>`_. Outside this
redshift range, we adopt a nearest-neighbor solution, i.e., the z = 0.2 and z =
3 mass functions.

2. ``"const_phi = False"``

The mass functions are switched to those from `tacchella18
<https://ui.adsabs.harvard.edu/abs/2018ApJ...868...92T/abstract>`_ between 4 < z
< 12, with the 3 < z < 4 transition from `leja20
<https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract>`_ managed with
a smoothly-varying average in number density space. We use the z = 12 mass
function for z > 12.


Galaxy Number Density
-----------

This prior informs the model about the survey volume being probed. It is
sensitive to the mass-completeness limit of the data. We provide a default
setting derived from a mock JWST catalog, which is contained in
``prior_data/mc_from_mocks.txt``.
In practice one would likely need to obtain
the mass-completeness limits from using SED-modeling heuristics based on the
flux-completeness limits in a given catalog.


Dynamic Star-formation History
-----------

The SFH is described non-parametrically as in Prospector-alpha; the number of
age bins is set by ``"nbins_sfh"``.

In contrast to the null expectation assumed in Prospector-alpha, the expectation
value in each age bin is matched to the cosmic star formation rate densities in
`behroozi19 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B/abstract>`_,
while the distribution about the mean remains to be the Student's-t
distribution. The sigma of the Student's-t distribution is set by
``"logsfr_ratio_tscale"``, and the range is clipped to be within
``"logsfr_ratio_mini"`` and ``"logsfr_ratio_maxi"``.

A simple mass dependence on SFH is further introduced by shifting the start of
the age bins as a function of mass. This SFH prior effectively encodes an
expectation that high-mass galaxies form earlier, and low-mass galaxies form
later, than naive expectations from the cosmic SFRD.


Stellar Mass-Stellar Metallicity
-----------

This is the stellar mass-stellar metallicity relationship measured from the SDSS
in `gallazzi05
<https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G/abstract>`_, introduced
in `leja19 <https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract>`_.
