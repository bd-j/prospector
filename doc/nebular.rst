Nebular Emission
================

Nebular emission in prospector is based on the implementation in FSPS, which
uses `cloudy <https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home>`_
photoionization predictions for emission lines and nebular continuum with FSPS
stellar populations as the ionization sources, as described in `byler17
<https://ui.adsabs.harvard.edu/abs/2017ApJ...840...44B/abstract>`_.

FSPS Nebular Emission Parameters
--------------------------------
The fundamental parameters of the nebular emission model are the ionization
parameter U (``"gas_logu"``) and the gas-phase metallicity (``"gas_logz"``). In
FSPS is possible to turn on or off nebular emission (line and continuum) using
the ``"add_neb_emission"`` switch.  It is also possible to turn off only the
nebular continuum with the ``"add_neb_continuum"`` switch. By default FSPS will
add the emission lines to the model spectrum internally.  In some situations
(see below) this is not desired, and so the ``"nebemlineinspec"`` switch can be
used to keep the lines from being added -- though their luminosities will still
be computed by FSPS -- in which case prospector can be used to add the lines to
the predicted spectra and photometry.

The nebular emission grids were computed for ionization sources (stellar
populations) with the same metallicity as the gas-phase, so to keep perfect
consistency it is necessary to tie these parameters together.


Fitting Emission Line Fluxes
--------------------------------------
The nebular emission line grids available in FSPS may not be flexible enough to
fit all the emission line ratios observed in nature (e.g. due to AGN actvitiy,
shocks, or radiative transfer, photoionization, and abundance effects not
captured in the cloudy modeling).  Nevertheless, it may be desirable to have
accurate models for the emission lines, e.g. to account for their presence in
the center of an absorption line of interest.  For this reason it is possible
with prospector, when fitting spectroscopic data, to *fit* for the true emission
line fluxes and compute model likelihoods marginalized over the possible
emission line fluxes.  The methodology is described in
`johnson21 <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_,
and is enabled via the ``"marginalize_elines"`` model parameter.

Briefly, explicit parameters are introduced to account for the emission line
widths (``"eline_sigma"``) and redshift offset from the stellar model
(``"eline_delta_zred"``). These default to 100 km/s and the systemic redshift
respectively, but can be adjusted or sampled and can be the same for every line
(scalar) or give the value separately for every line (vector). The maximum
likelihood emission line fluxes and the uncertainties thereon can then be
determined from linear least-squares.  Priors based on the cloudy-FSPS
predictions can also be incorporated.

Note that this yields emission line ratios that are no longer tied to a physical
model.  It is possible to use the cloudy-FSPS predictions of the line
luminosities for *some* lines while fitting for others, and this is described
below.

Choosing Lines to Fit, Fix, or Ignore
-------------------------------------
Several additional parameters can be used to decide which lines to fit
and marginalize over and which lines to include with luminosities set to the
cloudy-FSPS values.  It is also possible to completely ignore the contribution
of specific lines (i.e. to not include them in the modeled spectrum or
photometry).  These parameters are:

* ``"elines_to_fit"`` A list of lines to fit via linear-least-squares and
    marginalize over.  If this parameter is not given but
    ``"marginalize_elines"`` is True, all lines within the observed spectral
    range will be fit.
* ``"elines_to_fix"`` A list of lines to fix to their cloudy+FSPS predicted
    luminosities.
* ``"elines_to_ignore"`` A list of lines to ignore in the spectral and
    photometric models.  Thier luminosities are still computed, and can be
    accessed through the :py:class:`prospect.models.sedmodel.SpecModel` object
    at each likelihood call.  This parameter takes effect regardless of the
    presence of spectral data or the value of ``"marginalize_elines"``

In all cases the line names to use in these lists are those given in the FSPS
emission line line information table, ``$SPS_HOME/data/emlines_info.dat``, e.g.
``"Ly alpha 1216"`` for the Lyman-alpha 1216 Angstrom line.

Nebular Parameter Templates
---------------------------

Several default model parameter templates are available in
:py:class:`prospect.models.templates.TemplateLibrary` to easily include
different emission line treatments in prospector modeling including properly
setting the values of the various switches described above.

* ``"nebular"`` A basic parameter set in which the nebular emission is based
    on the cloudy-FSPS grids, the lines are added to the model within FSPS, and
    the gas-phase metallicity is tied to the stellar metallicity.  The only free
    parameter introduced by this template is ``"gas_logu"``.
* ``"nebular_marginalization"`` A parameter set for fitting and
    marginalizing over the emission line luminosities, with a prior based on the
    cloudy-fsps predictions.  By default all lines will be fit as long as FSPS
    is installed, and the line widths are included as a free parameter with
    uniform prior.  This parameter set adds one new free parameter,
    ``"eline_sigma"``.
* ``"fit_eline_redshift"`` This template can be used with
    ``"nebular_marginaliztion"`` above to also fit for the redshift offset of
    the emission lines from the stellar model.  It adds one free parameter,
    ``"eline_delta_zred"``.