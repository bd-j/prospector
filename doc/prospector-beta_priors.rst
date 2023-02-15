
Prospector-beta Priors
==============

This model is intended for fitting galaxy photometry where the redshift is unknown.
The priors encode empirical constraints of redshifts, masses, and star formation histories in the galaxy population.

A set of prospector parameters implementing the full set of priors is available as the ``"beta_nzsfh"`` entry
of :py:class:`prospect.models.templates.TemplateLibrary`.

We provide different combinations of the priors for flexibility. Specifically, we include the following:

* ``PhiMet``      : mass funtion + mass-met
* ``ZredMassMet`` : number density + mass funtion + mass-met
* ``PhiSFH``      : mass funtion + mass-met + SFH(M, z)
* ``NzSFH``       : number density + mass funtion + mass-met + SFH(M, z); this is the full set of Prospector-beta priors.

We describe each of the priors briefly below.

Stellar Mass Function
-----------

Two options are available, the choice of which depends on the given scientific question.
The relevant data files are ``prior_data/pdf_of_z_l20.txt`` & ``prior_data/pdf_of_z_l20t18.txt``.
These mass functions can also be replaced by supplying new data files to ``prior_data/``

1. ``"const_phi = True"``

The mass functions between 0.2 ≤ z ≤ 3.0 are taken from Leja et al. (2020). Outside this redshift range, we adopt a nearest-neighbor solution, i.e., the z = 0.2 and z = 3 mass functions.

2. ``"const_phi = False"``

The mass functions are switched to those from Tacchella et al. (2018) between 4 < z < 12, with the 3 < z < 4 transition from Leja et al. (2020) managed with a smoothly-varying average in number density space. We use the z = 12 mass function for z > 12.


Galaxy Number Density
-----------

This prior informs the model about the survey volume being probed. It is sensitive to the mass-completeness limit of the data. We provide a default setting derived from a mock JWST catalog, which is contained in ``prior_data/mc_from_mocks.txt``.
In practice one would likely need to obtain the mass-completeness limits from using SED-modeling heuristics based on the flux-completeness limits in a given catalog.


Dynamic Star-formation History
-----------

The expectation value in each age bin is matched to the cosmic star formation rate densities in Behroozi et al. (2019), while the distribution about the mean remains identical to the Student’s-t distribution in Prospector-alpha.

A simple mass dependence on SFH is further introduced by shifting the start of the age bins as a function of mass. This SFH prior effectively encodes an expectation that high-mass galaxies form earlier, and low-mass galaxies form later, than naive expectations from the cosmic SFRD.


Stellar Mass–Stellar Metallicity
-----------

This is the stellar mass–stellar metallicity relationship measured from the SDSS (Gallazzi et al. 2005), introduced in Leja et al. (2019).
