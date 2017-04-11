The Model
=========

Parameter specification
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

``"prior_function"``
    The prior function (e.g. ``tophat``)

``"prior_args"``
    A dictionary of arguments to the prior function (e.g. ``"mini":0, "maxi":100``)    

``"init_disp"``
    The dispersion in this parameter to use when generating an emcee sampler ball.  This is not technically required, as defaults 

Prior functions can be found in the ``priors`` module.
If you're using object priors it is also possible to replace the ``"prior_function"``  and ``"prior_args"`` keys with a single ``"prior"`` key with a value of, e.g. ``TopHat(mini=0, maxi=100)``.


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
