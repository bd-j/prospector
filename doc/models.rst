Models
======


Parameter Specification
------------------------------

All model parameters require a specification in the **parameter file**.
For a single parameter the specification is a dictionary that must at minimum include several keys:

``"name"``
    The name of the parameter, string.

``"N"``
    An integer specifying the length of the parameter.
    For the common case of a scalar parameter, use ``1``.

``"init"``
    The initial value of the parameter.
    If the parameter is free to vary, this is where optimization or will start from, or, if no optimization happens, this will be the center of the initial ball of `emcee` walkers.
    If using nested sampling then the value of ``"init"`` is not important (though a value must still be given.)
    If the parameter is not free, then this is the value that will be used throughout optimization and sampling.

``"isfree"``
    Boolean specifying whether a parameter is free to vary during
    optimization and sampling (``True``) or not (``False``).

For parameters with ``"isfree": True`` the following additional key is required:

``"prior"``
    An instance of a prior object, including parameters for the prior
    (e.g. ``priors.TopHat(mini=10, maxi=12)``).

If using ``emcee``, the following key can be useful to have:
    
``"init_disp"``
    The dispersion in this parameter to use when generating an ``emcee`` sampler ball.
    This is not technically required, as it defaults to 10% of the initial value.
    It is ignored if nested sampling is used.

It's also a good idea to have a ``"units"`` key, a string describing the units of the the parameter.
So, in the end, this looks something like

.. code-block:: python

    mass = {"name": "mass",
                  "N": 1,
                  "init": 1e9,
                  "init_disp": 1e8, # only important if using emcee sampling
                  "units": "M$_\odot$ of stars formed.",
                  "isfree": True,
                  "prior": priors.LogUniform(mini=1e7, maxi=1e12)
                  }

Nearly all parameters used by FSPS can become a model parameter.
When fitting galaxies the default python-FSPS parameter values will be used unless specified in a fixed parameter,
e.g. ``imf_type`` can be changed by including it as a fixed parameter with value given by ``"init"``.

Parameters can also be used to control the Prospector-specific parts of the modeling code.
These include things like spectral smoothing, wavelength calibration, spectrophotometric calibration, and any parameters of the noise model.
Be warned though, if you include a parameter that does not affect the model the code will not complain,
and if that parameter is free it will simply result in a posterior PDF that is the same as the prior (though optimization algorithms may fail).


Priors
---------

Prior objects can be found in the :py:mod:`prospect.models.priors` module.
It is recommended to use the objects instead of the functions,
as they have some useful attributes and are suitable for all types of sampling.
The prior functions by contrast will not work for nested sampling.
When specifying a prior using an object, you can and should specify the parameters of that prior on initialization, e.g.

.. code-block:: python

		mass["prior"] = priors.ClippedNormal(mean=0.0, sigma=1.0, mini=0.0, maxi=3.0)``


Parameter Set Templates
--------------------------------

A number of predefined sets of parameters (with priors) are available as
dictionaries of model specifications from ``models.templates.TemplateLibrary``,
these can be a good starting place for building your model.
To see the available parameter sets to inspect the free and fixed parameters in
a given set, you can do something like

.. code-block:: python
		
		from prospect.models.templates import TemplateLibrary
		# Show all pre-defined parameter sets
		TemplateLibrary.show_contents()
		# Show details on the "parameteric" set of parameters
		TemplateLibrary.describe("parametric_sfh")
		# Simply print all parameter specifications in "parametric_sfh"
		print(TemplateLibrary["parametric_sfh"])
		# Actually get a copy of one of the predefined sets
		model_params = TemplateLibrary["parametric_sfh"]



The ``load_model()`` Method
------------------------------------------

This method in the **parameter file** should take the ``run_params`` dictionary
as an argument list, and return an instance of the :class:`ProspectorParams`
subclass.

The :class:`ProspectorParams` is initialized with a list or dictionary (keyed
by parameter name) of each of the model parameter specifications described
above. If using a list, the order of the list sets the order of the free parameters in
the parameter vector.  The free parameters will be varied by the code during
the optimization and sampling phases.  The initial value from which
optimization is begun is set by the ``"init"`` values of each parameter.  For
fixed parameters the ``"init"`` value gives the value of that parameter to use
throughout the optimization and sampling phases (unless the ``"depends_on"``
key is present, see :doc:`advanced`.)

The ``run_params`` dictionary of arguments (including command line
modifications) can be used to modify the model parameters within this method
before the :class:`ProspectorParams` model object is instantiated.
