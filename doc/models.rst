The Model
=========

Parameter specification and Priors
-------------------------------

All model parameters require a specification in the **parameter file**.
For a single parameter the specification is a dictionary that must at minimum include several keys:

``"name"``
    The name of the parameter, string.

``"N"``
    An integer specifying the length of the parameter.
    For the common case of a scalar parameter, use ``1``.

``"init"``
    The initial value of the parameter.
    If the parameter is free to vary, this is where optimization will start from.
    If the parameter is not free, then this is the value that will be used throughout optimization and sampling.

``"isfree"``
    Boolean specifying whether a parameter is free to vary during
    optimization and sampling (``True``) or not (``False``).

For parameters with ``"isfree": True`` the following additional keys of the dictionary are required:

``"prior"``
    The prior object or function (e.g. ``priors.TopHat(mini=10, maxi=12)`` or ``priors.tophat``).
    The use of functions is deprecated.

``"prior_args"``
    This is only required if using prior functions (``priors.tophat``) instead
    of prior objects (``priors.TopHat``).
    It is a dictionary of arguments to the prior function (e.g. ``"mini":0, "maxi":100``)    

``"init_disp"``
    The dispersion in this parameter to use when generating an emcee sampler ball.
    This is not technically required, as it defaults to 10% of the initial value.
    It is ignored if nested sampling is used.

Prior functions and objects can be found in the ``priors`` module.
It is recemmended to use the objects instead of the functions,
as they have some useful attributes and are suitable for all types of sampling.
The prior functions by constrast will not work for nested sampling.
When specifiying a prior using an object, you can and should specify the parameters of that prior on initialization, e.g.
``priors.ClippedNormal(mean=0.0, sigma=1.0, mini=0.0, maxi=3.0)``

It's also a good idea to have a ``"units"`` key, a string describing the units of the the parameter.


The ``model_params`` List
-------------------------------------

This is simply a list of dictionaries describing the model parameters.
It is passed to the ``ProspectorParams`` object on initialization.
The free parameters will be varied by the code during the optimization and sampling phases.
The initial value from which optimization is begun is set by the ``"init"`` values of each parameter.
For fixed parameters the ``"init"`` value gives the value of that parameter to use throughout the optimization and MCMC phases
(unless the ``"depends_on"`` key is present, see Advanced_.)

Nearly all parameters used by FSPS can be set (or varied) here.
When fitting galaxies the default FSPS parameter values will be used unless specified in a fixed parameter,
e.g. ``imf_type`` can be changed by including it as a fixed parameter with value given by ``"init"``.
More generally any parameter used by the ``sources`` object to build an SED can be in the ``model_params`` list.


The ``load_model()`` method
------------------------------------------

This should return an instance of a subclass of the ``prospect.models.ProspectorParams`` object.
It is given the ``run_params`` dictionary as an argument list,
so the model can be modified based on keywords given there (or at the command line).


The ``load_sps()`` function
-------------------------------------

The likelihood function and SED models take an object (``sps``) from  ``prospect.sources`` as an argument.
This object should be returned by the ``load_sps()`` function in the **parameter file**.
The ``sps`` object generally includes all the spectral libraries necessary to build a model,
as well as some model building code.
This object is defined globally to enable multiprocessing, since generally it can't (or shouldn't) be serialized
and sent to other processors.


The ``load_gp()`` function
-------------------------------------

This function should return a NoiseModel object for the spectroscopy or photometry.
Either or both can be ``None`` in which case the likelihood will not include covariant noise.
