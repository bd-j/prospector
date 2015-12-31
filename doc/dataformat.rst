Data Formats
===========

The ``obs`` Dictionary
--------------------------------

|Codename| expects the data in the form of a dictionary.
This dictionary should have (at least) the following keys and values:

-  ``"wavelengths"``: The wavelength vector for the spectrum, ndarray.

- ``"spectrum"``: The flux vector for the spectrum,
  ndarray of same length as the wavelength vector.

- ``"unc"``: The uncertainty vector, in units of ``"spectrum"``,
  ndarray of same length as the wavelength vector.

- ``"mask"``: A boolean array of same length as the wavelength vector,
  where ``False`` elements are ignored in the likelihood calculation.

- ``"filters"``: A sequence of `sedpy<https://github.com/bd-j/sedpy>`_ filter objects or filter names,
  used to calculate model magnitudes.

- ``"maggies"``: An array of *maggies*, same length as ``"filters"``.
  Maggies are a linear flux unit defined as :math:`maggie = 10^{-0.4m_{AB}}`

-  ``"maggies_unc"``: An array of photometric uncertainties, same length as ``"filters"``,
   that gives the photometric uncertainties in units of *maggies*

-  ``"phot_mask"``: Like ``"mask"``, a boolean array, used to mask the
   photometric data during the likelihood calculation.

If you do not have spectral or photometric data, set ``"spectrum"=None`` or
``"maggies"=None`` respectively.

The ``load_obs()`` function
---------------------------------------

The ``load_obs(**kwargs)`` function in the parameter file should take the ``run_params`` dictionary (modified by command line arguments) as keyword arguments and return an ``obs`` dictionary described above.
