# -*- coding: utf-8 -*-

import json
import numpy as np

__all__ = ["Observation", "Spectrum", "Photometry",
           "from_oldstyle"]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class Observation:

    logify_spectrum = False
    alias = {}

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 **kwargs
                 ):

        self.flux = flux
        self.uncertainty = uncertainty
        self.mask = mask
        self.from_oldstyle(**kwargs)

    def __getitem__(self, item):
        """Dict-like interface for backwards compatibility
        """
        k = self.alias.get(item, item)
        return getattr(self, k)

    def get(self, item, default):
        try:
            return self[item]
        except(AttributeError):
            return default

    def from_oldstyle(self, **kwargs):
        """Take an old-style obs dict and use it to populate the relevant
        attributes.
        """
        for k, v in self.alias.items():
            if k in kwargs:
                setattr(self, v, kwargs[k])

    def rectify(self):
        """Make sure required attributes are present and have the appropriate
        sizes.  Also auto-masks non-finite data or negative uncertainties.
        """
        assert self.wavelength.ndim == 1, "`wavelength` is not 1-d array"
        assert self.ndata > 0, "no wavelength points supplied!"
        assert len(self.wavelength) == len(self.flux), "Flux array not same shape as wavelength"
        assert len(self.wavelength) == len(self.uncertainty), "Uncertainty array not same shape as wavelength"

        # make mask array with automatic filters
        marr = np.zeros(self.ndata, dtype=bool)
        marr[self.mask] = True
        self.mask = (marr &
                     (np.isfinite(self.flux)) &
                     (np.isfinite(self.uncertainty)) &
                     (self.uncertainty > 0))

        assert self.ndof > 0, "No valid data to fit: check the sign of the masks."

    def render(self, wavelength, spectrum):
        raise(NotImplementedError)

    @property
    def ndof(self):
        return int(self.mask.sum())

    @property
    def ndata(self):
        if self.wavelength is None:
            return 0
        else:
            return len(self.wavelength)

    def serialize(self):
        obs = vars(self)
        serial = json.dumps(obs, cls=NumpyEncoder)


class Photometry(Observation):

    kind = "photometry"
    alias = dict(maggies="flux",
                 maggies_unc="uncertainty",
                 filters="filters",
                 phot_mask="mask")

    def __init__(self, filters=[], **kwargs):

        super(Photometry, self).__init__(**kwargs)
        self.filters = filters

    def render(self, wavelength, spectrum):
        w, s = wavelength, spectrum
        mags = [f.ab_mag(w, s, **self.render_kwargs)
                for f in self.filters]
        return 10**(-0.4 * np.array(mags))

    @property
    def wavelength(self):
        return np.array([f.wave_effective for f in self.filters])

    def to_oldstyle(self):
        obs = vars(self)
        obs.update({k: self[v] for k, v in self.alias.items()})
        _ = [obs.pop(k) for k in ["flux", "uncertainty", "mask"]]
        obs["phot_wave"] = self.wavelength
        return obs


class Spectrum(Observation):

    kind = "spectrum"
    alias = dict(spectrum="flux",
                 unc="uncertainty",
                 wavelength="wavelength",
                 mask="mask")

    def __init__(self,
                 wavelength=None,
                 resolution=None,
                 calibration=None,
                 **kwargs):

        """
        :param resolution: (optional, default: None)
            Instrumental resolution at each wavelength point in units of km/s
            dispersion (:math:`= c \, \sigma_\lambda / \lambda = c \, \FWHM_\lambda / 2.355 / \lambda = c / (2.355 \, R_\lambda)`
            where :math:`c=2.998e5 {\rm km}/{\rm s}`

        :param calibration:
            not sure yet ....
        """
        super(Spectrum, self).__init__(**kwargs)
        self.wavelength = wavelength
        self.resolution = resolution
        self.calibration = calibration

    def render(self, wavelength, spectrum):
        if self.ndata > 0:
            wave = self.wavelength
            spec = np.interp(wave, wavelength, spectrum)
        return wave, spec

    def to_oldstyle(self):
        obs = vars(self)
        obs.update({k: self[v] for k, v in self.alias.items()})
        _ = [obs.pop(k) for k in ["flux", "uncertainty"]]
        return obs


def from_oldstyle(obs):
    """Convert from an oldstyle dictionary to a list of observations
    """
    return [Spectrum().from_oldstyle(obs), Photometry().from_oldstyle(obs)]