Models
======

The modular nature of |Codename| allows it to be applied to a variety of data
types and different scientific questions.  However, this flexibility requires
the user to take care in defining the model, including prior distributions,
which may be specific to a particular scientific question or to the data to be
fit. Different models and prior beliefs may be more or less appropriate for
different kinds of galaxies.  Certain kinds of data may require particular model
components in order to be described well. |Codename| strives to make this model
building process as straightforward as possible, but it is a step that cannot be
skipped.


The choice of which parameters to include, which to let vary, and what prior
distributions to use will depend on the data being fit (including the types of
objects) and the goal of the inference.  As examples, if no infrared data is
available then it is not necessary to fit  -- or even include in the model
at all -- the dust emission parameters controlling the shape of the infrared
SED.  For globular clusters or completely quenched galaxies it may not be
necessary to include nebular emission in the model, and one may wish to adjust
the priors on population age or star formation history to be more appropriate
for such objects.  If spectroscopic data is being fit then it may be necessary
to include velocity dispersion as a free parameter.  Generating and fitting mock
data can be an extremely useful tool for exploring the sensitivity of a given
type of data to various parameters.


Parameter Specification
------------------------------

A model is defined by a dictionary of parameter specifications, keyed by
parameter name, that is used to instantiate and configure the model objects
(instances of :py:class:`models.ProspectorParams` or its subclasses.) This
dictionary is usually constructed or given in a **parameter file**.

For a single parameter the specification is a dictionary that should at minimum
include several keys:

``"N"``
    An integer specifying the length of the parameter.
    If not supplied this defaults to ``1``, the common case of a scalar parameter.

``"isfree"``
    Boolean specifying whether a parameter is free to vary during optimization
    and sampling (``True``) or not (``False``). This defaults to ``False`` if
    not supplied.

``"init"``
    The initial value of the parameter.
    If the parameter is not free, then this is the value that will be used
    throughout optimization and sampling.
    If the parameter is free to vary, this is where optimization will start
    from or -- if no optimization happens -- this will be the center of the initial
    ball of `emcee` walkers. Note that if using nested sampling then the value of
    ``"init"`` is not important (though a value must still be given).

For parameters with ``isfree=True`` the following additional key is required:

``"prior"``
    An instance of a prior object, including parameters for the prior
    (e.g. ``priors.TopHat(mini=10, maxi=12)``).

If using ``emcee``, the following key can be useful to have:

``"init_disp"``
    The dispersion in this parameter to use when generating an ``emcee`` sampler ball.
    This is not technically required, as it defaults to 10% of the initial value.
    It is ignored if nested sampling is used.

It's also a good idea to have a ``"units"`` key, a string describing the units of the the parameter.
So, in the end, this looks something like:

.. code-block:: python

    mass = dict(N=1, init=1e9, isfree=True,
                prior= priors.LogUniform(mini=1e7, maxi=1e12),
                units="M$_\odot$ of stars formed.", init_disp=1e8)
    model_params = dict(mass=mass)

Nearly all parameters used by FSPS can become a model parameter. When fitting
galaxies the default python-FSPS parameter values will be used unless specified
in a fixed parameter, e.g. ``imf_type`` can be changed by including it as a
fixed parameter with value given by ``"init"``.

Parameters can also be used to control the Prospector-specific parts of the
modeling code. These include things like spectral smoothing, wavelength
calibration, spectrophotometric calibration, and any parameters of the noise
model. Be warned though, if you include a parameter that does not affect the
model the code will not complain, and if that parameter is free it will simply
result in a posterior PDF that is the same as the prior (though optimization
algorithms may fail).


Priors
------

All parameters that are free to vary must have an associated prior distribution.
Prior objects can be found in the :py:mod:`prospect.models.priors` module. When
specifying a prior using an object, you can and should specify the parameters of
that prior on initialization, e.g.

.. code-block:: python

		model_params["dust2"]["prior"] = priors.ClippedNormal(mean=0.3, sigma=0.5, mini=0.0, maxi=3.0)


Transformations
---------------

Sometimes the native parameterization of stellar population models is not the
most useful.  In these cases parameter *transformations* can prove useful.

Transformations are useful to impose parameter limits that are a function of
other parameters; for example, when fitting for redshift it can be useful to
reparameterize the age of a population (say, in Gyr) into its age as a fraction
of the age of the universe at that redshift.  This avoids the problem of
populations that are older than the age of the universe, or complicated joint
priors on the population age and the redshift.  A number of useful
transformation functions are provided in |Codename| and these may be easily
supplemented with user defined functions.

This parameter transformation and dependency mechanism can be used to tie any
number of parameters to any number of other parameters in the model, as long as
the latter parameters are not *also* dependent on some parameter transformation.
This mechanism may also be used to avoid joint priors.  For example, if one
wishes to place a prior on the ratio of two parameters (say, that it be less
than one) then the ratio itself can be introduced as a new parameter, and one of
the original parameters can be "fixed" but have its value at each parameter
location depend on the other original parameter and the new ratio parameter.

As a simple example, we consider sampling in the log of the SF timescale instead
of the timescale itself.  The follwing code

.. code-block:: python

    def delogify(logtau=0, **extras):
    	return 10**logtau

    model_params["tau"]["isfree"] = False
    model_params["tau"]["depends_on"] = delogify
    model_params["logtau"] = dict(N=1, init=0, isfree=True, prior=priors.TopHat(mini=-1, maxi=1))


could be used to set the value of ``tau`` using the free parameter ``logtau``
(i.e., sample in the log of a parameter, though setting a :py:class:`prospect.models.priors.LogUniform`
prior is equivalent in terms of the posterior).

This dependency function must take optional extra keywords (``**extras``)
because the entire parameter dictionary will be passed to it. Then add the new
parameter specification to the ``model_params`` dictionary for the parameter
that can vary (and upon which the fixed parameter depends). In this example that
new free parameter would be ``logtau``.

This pattern can also be used to tie arbitrary parameters together (e.g.
gas-phase and stellar metallicity) while still allowing them to vary. A
parameter may depend on multiple other (free or fixed) parameters, and multiple
parameters may depend on a single other (free or fixed) parameter.  This
mechanism is used extensively for the non-parametric SFHs, and is recommended
for complex dust attenuation models.

.. note::
    It is important that any parameter with the ``"depends_on"`` key present is a
    fixed parameter. For portability and easy reconstruction of the model it is
    important that the ``depends_on`` function either be importable (e.g. one of the
    functions supplied in :py:mod:`prospect.models.transforms`) or defined within
    the parameter file.


Parameter Set Templates
--------------------------------

A number of predefined sets of parameters (with priors) are available as
dictionaries of model specifications from
:py:class:`prospect.models.templates.TemplateLibrary`, these can be a good
starting place for building your model. To see the available parameter sets to
inspect the free and fixed parameters in a given set, you can do something like

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
        # This dictionary can be updated or modified, to expand the model.
        model_params.update(TemplateLibrary["nebular"])
		# Instantiate a model object
		from prospect.models import SpecModel
		model = SpecModel(model_params)


The ``build_model()`` Method
------------------------------------------

This method in the **parameter file** should take the ``run_params`` dictionary
as keyword arguments, and return an instance of a subclass of
:py:class:`prospect.models.ProspectorParams`.

The model object, a subclass of :py:class:`prospect.models.ProspectorParams`, is
initialized with a list or dictionary (keyed by parameter name) of each of the
model parameter specifications described above. If using a list, the order of
the list sets the order of the free parameters in the parameter vector.  The
free parameters will be varied by the code during the optimization and sampling
phases.  The initial value from which optimization is begun is set by the
``"init"`` values of each parameter.  For fixed parameters the ``"init"`` value
gives the value of that parameter to use throughout the optimization and
sampling phases (unless the ``"depends_on"`` key is present, see
:doc:`advanced`.)

The ``run_params`` dictionary of arguments (including command line
modifications) can be used to change how the model parameters are specified
within this method before the :py:class:`prospect.models.ProspectorParams` model
object is instantiated. For example, the value of a fixed parameter like
``zred`` can be set based on values in ``run_params`` or additional parameters
related to dust or nebular emission can be optionally added based on switches in
``run_params``.

Useful model objects include :py:class:`prospect.models.SpecModel` and
:py:class:`prospect.models.PolySpecModel`. The latter includes tools for
optimization of spectrophotometric calibration.



.. |Codename| replace:: Prospector
