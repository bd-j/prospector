Data Formats
===========
The :py:meth:`load_obs()` function
---------------------------------------

The :py:meth:`load_obs(**kwargs)` function in the parameter file is written by the user and should take the ``run_params`` dictionary (modified by command line arguments) as keyword arguments.
It should return an ``obs`` dictionary described below.

Other than that, the contents can be anything.
Within this function you might open and read FITS files, ascii tables, HDF5 files, or query SQL databases.
You could, using say an ``objid`` parameter, dynamically load data (including filter sets) for different objects in a table.
Feel free to import helper functions, modules, and packages (like astropy, h5py, sqlite, etc.)

The point of this function is that you don't have to externally coerce your data format to be what |Codename| expects and keep another version of files lying around: the coercion happens *within* the code itself.
Again, the only requirement is that the function can take a ``run_params`` dictionary as keyword arguments
and that it return an ``obs`` dictionary as described below.

The ``obs`` Dictionary & Data Units
--------------------------------

|Codename| expects the data in the form of a dictionary.
This dictionary should have (at least) the following keys and values:

``"wavelength"``
    The wavelength vector for the spectrum, ndarray.
    Units are vacuum Angstroms.
    The model spectrum will be computed for each element of this vector.
    Set to ``None`` if you have no spectrum.
    If fitting observed frame photometry as well,
    then these should be observed frame wavelengths.

``"spectrum"``
    The flux vector for the spectrum,
    ndarray of same length as the wavelength vector.
    If absolute spectrophotometry is available,
    the units of this spectrum should be Janskies divided by 3631 (i.e. maggies).
    Also the ``rescale_spectrum`` run parameter should be False.

``"unc"``
    The uncertainty vector (sigma), in same units as ``"spectrum"``,
    ndarray of same length as the wavelength vector.

``"mask"``
   A boolean array of same length as the wavelength vector,
   where ``False`` elements are ignored in the likelihood calculation.

``"filters"``
   A sequence of `sedpy <https://github.com/bd-j/sedpy>`_ filter objects or filter names,
   used to calculate model magnitudes.

``"maggies"``
    An array of *maggies*, same length as ``"filters"``.
    Maggies are a linear flux density unit defined as :math:`maggie = 10^{-0.4m_{AB}}`
    where :math:`m_{AB}` is the AB apparent magnitude.
    That is, 1 maggie is the flux density in Janskys divided by 3631.
    Set to ``None`` if you have no photometric data.

``"maggies_unc"``
    An array of photometric uncertainties, same length as ``"filters"``,
    that gives the photometric uncertainties in units of *maggies*

``"phot_mask"``
    Like ``"mask"``, a boolean array, used to mask the
    photometric data during the likelihood calculation.

If you do not have spectral or photometric data, set ``"wavelength": None`` or ``"maggies": None`` respectively.
Feel free to add keys that store other metadata, these will be stored on output.
However, for ease of storage these keys should either be numpy arrays or basic python datatypes that are JSON serializable
(e.g. strings, ints, and floats and lists, dicts, and tuples thereof.)

.. |Codename| replace:: Prospector
