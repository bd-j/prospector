# -*- coding: utf-8 -*-

import numpy as np


class ObsDict:

    def __init__(self):
        pass

    def rectify(self):
        pass


class Observation:

    def __init__(self,
                 flux=np.array([]),
                 uncertainty=np.array([]),
                 mask=np.array([], dtype=bool)
                 ):
        self.flux = flux
        self.uncertainty = uncertainty
        self.mask = mask

    def __getitem__(self, item):
        """Dict-like interface for backwards compatibility
        """
        k = self.alias.get(item)
        return getattr(self, k)

    def render(self, wavelength, spectrum):
        raise(NotImplementedError)

    @property
    def ndof(self):
        return self.mask.sum()


class Photometry(Observation):

    alias = {"maggies": "flux",
             "maggies_unc": "uncertainty"}

    def __init__(self, filters=[], **kwargs):

        super(Photometry, self).__init__(**kwargs)
        self.filters = filters

    def render(self, wavelength, spectrum):
        w, s = wavelength, spectrum
        mags = [f.ab_mag(w, s, **self.render_kwargs)
                for f in self.filters]
        return 10**(-0.4 * np.array(mags))

    @property
    def wavelengths(self):
        return np.array([f.wave_effective for f in self.filters])


class Spectrum(Observation):

    def __init__(self,
                 wavelength,
                 resolution=None,
                 calibration=None,
                 **kwargs):

        super(Spectrum, self).__init__(**kwargs)
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.calibration = calibration

    def render(self, wavelength, spectrum):
        w, s = wavelength, spectrum
        return s

if __name__ == "__main__":
    pass