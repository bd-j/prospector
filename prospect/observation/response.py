# -*- coding: utf-8 -*-

import json
import numpy as np

from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.interpolate import splrep, BSpline
from scipy.signal import medfilt

from .observation import wave_to_x


__all__ = ["SplineOptCal", "PolyOptCal"]


CKMS = 2.998e5


class PolyOptCal:

    """A mixin class that allows for optimization of a Chebyshev response
    function given a model spectrum.
    """

    def __init__(self, *args,
                 polynomial_order=0,
                 polynomial_regularization=0,
                 median_polynomial=0,
                 **kwargs):
        super(PolyOptCal, self).__init__(*args, **kwargs)
        self.polynomial_order = polynomial_order
        self.polynomial_regularization = polynomial_regularization
        self.median_polynomial = median_polynomial

    def _available_parameters(self):
        # These should both be attached to the Observation instance as attributes
        pars = [("polynomial_order", "order of the polynomial to fit"),
                ("polynomial_regularization", "vector of length `polyorder` providing regularization for each polynomial term"),
                ("median_polynomial", "if > 0, median smooth with a kernel of width order/range/median_polynomial before fitting")
                ]

        return pars

    def compute_response(self, spec=None, extra_mask=True, **kwargs):
        """Implements a Chebyshev polynomial response function model. This uses
        least-squares to find the maximum-likelihood Chebyshev polynomial of a
        certain order describing the ratio of the observed spectrum to the model
        spectrum. If emission lines are being marginalized out, they should be
        excluded from the least-squares fit using the ``extra_mask`` keyword

        Parameters
        ----------
        spec : ndarray of shape (Spectrum().ndata,)
            The model spectrum on the observed wavelength grid

        extra_mask : ndarray of Booleans of shape (Spectrum().ndata,)
            The extra mask to be applied.  True=use data, False=mask data

        Returns
        -------
        response : ndarray of shape (nwave,)
           A polynomial given by :math:`\sum_{m=0}^M a_{m} * T_m(x)`.
        """

        order = self.polynomial_order
        reg = self.polynomial_regularization
        mask = self.mask & extra_mask
        assert (self.mask.sum() > order), f"Not enough points to constrain polynomial of order {order}"

        polyopt = (order > 0)
        if (not polyopt):
            print("no polynomial")
            self.response = np.ones_like(self.wavelength)
            return self.response

        x = wave_to_x(self.wavelength, mask)
        y = (self.flux / spec - 1.0)[mask]
        yerr = (self.uncertainty / spec)[mask]
        yvar = yerr**2

        if self.median_polynomial > 0:
            kernel_factor = self.median_polynomial
            knl = int((x.max() - x.min()) / order / kernel_factor)
            knl += int((knl % 2) == 0)
            y = medfilt(y, knl)

        Afull = chebvander(x, order)
        A = Afull[mask, :]
        ATA = np.dot(A.T, A / yvar[:, None])
        if np.any(reg > 0):
            ATA += reg**2 * np.eye(order+1)
        c = np.linalg.solve(ATA, np.dot(A.T, y / yvar))

        poly = np.dot(Afull, c)

        self._chebyshev_coefficients = c
        self.response = poly + 1.0
        return self.response


class SplineOptCal:

    """A mixin class that allows for optimization of a Chebyshev response
    function given a model spectrum.
    """


    def __init__(self, *args,
                 spline_knot_wave=None,
                 spline_knot_spacing=None,
                 spline_knot_n=None,
                 **kwargs):
        super(SplineOptCal, self).__init__(*args, **kwargs)

        self.params = {}
        if spline_knot_wave is not None:
            self.params["spline_knot_wave"] = spline_knot_wave
        elif spline_knot_spacing is not None:
            self.params["spline_knot_spacing"] = spline_knot_spacing
        elif spline_knot_n is not None:
            self.params["spline_knot_n"] = spline_knot_n

        # build and cache the knots
        w = self.wavelength[self.mask]
        (wlo, whi) = w.min(), w.min()
        self.wave_x = 2.0 * (self.wavelength - wlo) / (whi - wlo) - 1.0
        self.knots_x = self.make_knots(wlo, whi, as_x=True, **self.params)


    def _available_parameters(self):
        pars = [("spline_knot_wave", "vector of wavelengths for the location of the spline knots"),
                ("spline_knot_spacing", "spacing between knots, in units of wavelength"),
                ("spline_knot_n", "number of interior knots between minimum and maximum unmasked wavelength")
                ]

        return pars

    def compute_response(self, spec=None, extra_mask=True, **extras):
        """Implements a spline response function model.  This fits a cubic
        spline with determined knot locations to the ratio of the observed
        spectrum to the model spectrum.  If emission lines are being
        marginalized out, they are excluded from the least-squares fit.

        The knot locations must be specified as model parameters, either
        explicitly or via a number of knots or knot spacing (in angstroms)
        """
        mask = self.mask & extra_mask


        splineopt = True
        if ~splineopt:
            self.response = np.ones_like(self.wavelength)
            return self.response

        y = (self.flux / spec - 1.0)[mask]
        yerr = (self.uncertainty / spec)[mask] # HACK
        tck = splrep(self.wave_x[mask], y[mask], w=1/yerr[mask], k=3, task=-1, t=self.knots_x)
        self._spline_coeffs = tck
        spl = BSpline(*tck)
        spline = spl(self.wave_x)

        self.response = (1.0 + spline)
        return self.response

    def make_knots(self, wlo, whi, as_x=True, **params):
        """Can we move this to instantiation?
        """
        if "spline_knot_wave" in params:
            knots = np.squeeze(params["spline_knot_wave"])
        elif "spline_knot_spacing" in params:
            s = np.squeeze(params["spline_knot_spacing"])
            # we drop the start so knots are interior
            knots = np.arange(wlo, whi, s)[1:]
        elif "spline_knot_n" in params:
            n = np.squeeze(params["spline_knot_n"])
            # we need to drop the endpoints so knots are interior
            knots = np.linspace(wlo, whi, n)[1:-1]
        else:
            raise KeyError("No valid spline knot specification in self.params")

        if as_x:
            knots = 2.0 * (knots - wlo) / (whi - wlo) - 1.0

        return knots


class PolyFitCal:

    """This is a mixin class that generates the
    multiplicative response vector as a Chebyshev polynomial described by the
    ``poly_param_name`` parameter of the model, which may be free (fittable)
    """

    def __init__(self, *args, poly_param_name=None, **kwargs):
        super(SplineOptCal, self).__init(*args, **kwargs)
        self.poly_param_name = poly_param_name

    def _available_parameters(self):
        pars = [(self.poly_param_name, "vector of polynomial chabyshev coefficients")]

        return pars

    def compute_response(self, **kwargs):
        """Implements a Chebyshev polynomial calibration model.  This only
        occurs if ``"poly_coeffs"`` is present in the :py:attr:`params`
        dictionary, otherwise the value of ``params["spec_norm"]`` is returned.

        :param theta: (optional)
            If given, set :py:attr:`params` using this vector before
            calculating the calibration polynomial. ndarray of shape
            ``(ndim,)``

        :param obs:
            A dictionary of observational data, must contain the key
            ``"wavelength"``

        :returns cal:
           If ``params["cal_type"]`` is ``"poly"``, a polynomial given by
           :math:`\times (\Sum_{m=0}^M```'poly_coeffs'[m]``:math:` \times T_n(x))`.
        """

        if self.poly_param_name in kwargs:
            mask = self.get('mask', slice(None))
            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = wave_to_x(self.wavelength, mask)
            # get coefficients.
            c = kwargs[self.poly_param_name]
            poly = chebval(x, c)
        else:
            poly = 1.0

        self.response = poly
        return self.response
