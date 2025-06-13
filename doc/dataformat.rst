Data Formats
============


The `Observation` class
-----------------------------------

|Codename| expects the data in the form of list of ``Observations``, preferably
returned by :py:meth:`build_obs` (see below). Each Observation instance
corresponds to a single dataset, and is basically a namespace that also supports
dict-like accessing of important attributes.  In addition to holding data and
uncertainties thereon, they tell |Codename| what data to predict, contain
dataset-specific information for how to predict that data, and can even store
methods for computing likelihoods in the case of complicated, dataset-specific
noise models.

There are two fundamental kinds of data, :py:class:`Photometry` and
:py:class:`Spectrum` that are each sub-classes of :py:class:`Observation`. There
is also also a :py:class:`Lines` class for integrated emission line fluxes. They
have the following attributes, most of which can also be accessed as dictionary
keys:


- ``wavelength``
    The wavelength vector for a `Spectrum`` or the effective wavelengths of the
    filters in a `Photometry` data set, ndarray. Units are vacuum Angstroms.
    Generally these should be observed frame wavelengths.

- ``flux``
    The flux vector for a :py:class:`Spectrum`, or the broadband fluxes for
    :py:class:`Photometry` ndarray of same length as the wavelength vector.

    For `Photometry` the units are *maggies*. Maggies are a linear flux density
    unit  defined as :math:`{\rm maggie} = 10^{-0.4 \, m_{AB}}` where
    :math:`m_{AB}` is the AB apparent magnitude. That is, 1 maggie is the flux
    density in Janskys divided by 3631. If absolute spectrophotometry is
    available, the units for a :py:class:`Spectrum`` should also be maggies,
    otherwise photometry must be present and a calibration vector must be
    supplied or fit.  Note that for convenience there is a `maggies_to_nJy`
    attribute of `Observation` that gives the conversion factor. For
    :py:class:`Lines`, the units should be erg/s/cm^2

- ``uncertainty``
    The uncertainty vector (sigma), in same units as ``flux``, ndarray of same
    length as the wavelength vector.

- ``mask``
    A boolean array of same length as the wavelength vector, where ``False``
    elements are ignored in the likelihood calculation.

- ``filters``
    For a `Photometry`, this is a list of strings corresponding to filter names
    in `sedpy <https://github.com/bd-j/sedpy>`_

- ``line_ind``
    For a `Lines` instance, the (zero-based) index of the emission line in the
    FSPS emission line
    `table <https://github.com/cconroy20/fsps/blob/master/data/emlines_info.dat>`_

In addition to these attributes, several additional aspects of an observation
are used to help predict data or to compute likelihoods.  The latter is
particularly important in the case of complicated noise models, including outlier
models, jitter terms, or covariant noise.

- ``name``
    A string that can be used to identify the dataset.  This can be useful for
    dataset-specfic parameters.  By default the name is constructed from the
    `kind` and the memory location of the object.

- ``resolution``
    For a `Spectrum` this defines the instrumental resolution.  Analagously to
    the ``filters`` attribute for `Photometry`, this knowledge is used to
    accurately predict the model in the space of the data.

- ``noise``
    A :py:class:`NoiseModel` instance.  By default this implements a simple
    chi-square calculation of independent noise, but it can be complexified.


Example
-------

For a single observation, you might do something like:

.. code-block:: python

        def build_obs(N):
            from prospect.observation import Spectrum
            N = 1000  # number of wavelength points
            spec = Spectrum(wavelength=np.linspace(3000, 5000, N), flux=np.zeros(N), uncertainty=np.ones(N))
            # ensure that this is a valid observation for fitting
            spec = spec.rectify()
            observations = [spec]

            return observations

Note that `build_obs` returns a *list* even if there is only one dataset.

For photometry this might look like:

.. code-block:: python

        def build_obs(N):
            from prospect.observation import Photometry
            # valid sedpy filter names
            fnames = list([f"sdss_{b}0" for b in "ugriz"])
            Nf = len(fnames)
            phot = [Photometry(filters=fnames, flux=np.ones(Nf), uncertainty=np.ones(Nf)/10)]
            # ensure that this is a valid observation for fitting
            phot = phot.rectify()
            observations = [phot]
            return observations

Converting from old style obs dictionaries
------------------------------------------

A tool exists to convert old combined observation dictionaries to a list of
`Observation` instances:

.. code-block:: python

        from prospect.observation import from_oldstyle
        # dummy observation dictionary with just a spectrum
        N = 1000
        obs = dict(wavelength=np.linspace(3000, 5000, N), spectrum=np.zeros(N), unc=np.ones(N),
                   filters=[f"sdss_{b}0" for b in "ugriz"], maggies=np.zeros(5), maggies_unc=np.ones(5))
        # ensure that this is a valid observation for fitting
        spec, phot = from_oldstyle(obs)
        print(spec.ndata, phot.filternames, phot.wavelength, phot.flux)



The :py:meth:`build_obs` function
---------------------------------

The :py:meth:`build_obs` function in the parameter file is written by the user.
It should take a dictionary of command line arguments as keyword arguments. It
should return a list of :py:class:`prospect.observation.Observation` instances,
described above.

Other than that, the contents can be anything. Within this function you might
open and read FITS files, ascii tables, HDF5 files, or query SQL databases. You
could, using e.g. an ``objid`` parameter, dynamically load data (including
filter sets) for different objects in a table. Feel free to import helper
functions, modules, and packages (like astropy, h5py, sqlite, astroquery, etc.)

The point of this function is that you don't have to *externally* convert your
data format to be what |Codename| expects and keep another version of files
lying around: the conversion happens *within* the code itself. Again, the only
requirement is that the function can take a ``run_params`` dictionary as keyword
arguments and that it return :py:class:`prospect.observation.Observation` instances, as
 described above.  Each observation instance should correspond to a particular
 dataset (e.g. a broadband photomtric SED, the spectrum from a particular
 instrument, or the spectrum from a particular night) that shares instrumental
 and, more importantly, calibration parameters.


.. |Codename| replace:: Prospector
