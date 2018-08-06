prospect.sources
============
Classes in the :py:mod:`prospect.sources` module are used to instantiate
**sps** objects.  They are defined by the presence of a :py:meth:`get_spectrum`
method that takes a wavelength array, a list of filter objects, and a parameter
dictionary and return a spectrum, a set of broadband fluxes, and a blob of
ancillary information.

Most of these classes are a wrapper on ``fsps.StellarPopulation`` objects, and
as such have a significant memory footprint.  The parameter dictionary can
include any ``fsps`` parameter, as well as parameters used by these classes to
control redshifting, spectral smoothing, wavelength calibration, and other
aspects of the model.

.. automodule:: prospect.sources
   :members: SSPBasis, CSPSpecBasis, FastSSPBasis, FastStepBasis, BlackBodyDustBasis
