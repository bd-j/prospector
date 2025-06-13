Noise Modeling
============

The noise model for each dataset is used to define the likelihood function,
given the observed data and the model prediction.  As such, each dataset or
``Observation`` is assigned its own noise model.  By default this is the basic
:math:`\chi^2` noise model.  Complications are described below


Outliers
--------

For outlier modeling we follow `hogg10 <https://arxiv.org/abs/1008.4686>`_

The key parameters of this noise model are the fraction of datapoints in a given
dataset that are outliers, and the typical variance of the outliers.  Each
dataset might have a different outlier parameters, and so we need to find a way
to identify which outlier model parameter belongs with which dataset.  This can
be done when the noise model is assigned to a dataset.  For example, if we had a
single photometric dataset and a single spectroscopic dataset, with outlier
model parameters for each given by ``("f_outlier_phot", "nsigma_outlier_phot")``
and ``("f_outlier_spec", "nsigma_outlier_spec")``  respectively (this is the
default parameter set available as a template) then we could associate these
parameter with each dataset as follows:

.. code-block:: python

    from prospect.data import Photometry, Spectrum
    from prospect.likelihood import NoiseModel
    filternames = [f"sdss_{b}0" for b in "ugriz"]
    N = len(fnames)
    pdat = Photometry(filters=filternames, flux=np.zeros(N), uncertainty=np.ones(N),
                      noise=NoiseModel(frac_out_name="f_outlier_phot",
                                       nsigma_out_name="nsigma_outlier_phot"))
    N = 1000
    sdat = Spectrum(wavelength=np.linspace(4e3, 7e3, N), np.zeros(N), np.ones(N),
                    noise=NoiseModel(frac_out_name="f_outlier_spec",
                                     nsigma_out_name="nsigma_outlier_spec"))

This can be combined with other Noise models, as long as they have diagonal
(1-dimensional) covariance matices.


Jitter
------



Correlated Noise
----------------


KDE Noise
---------



.. |Codename| replace:: Prospector
