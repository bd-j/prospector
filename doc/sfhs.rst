SFHs
====

Numerous star formation history (SFH) treatments are available in prospector.
Some of these are described blow, along with instructions for their use.

SSPs
----

Parametric SFH
--------------
So called "parametric" SFHs describe the SFR as a function of time via a
relatively simple function with just a few parameters.  In prospector the
parametric SFH treatment is actually handled by FSPS itself, and so the model
parameters requirted are the same as those in FSPS (see documentation)

The available parametric SFHs include constant, exponential decay (tau models),
and delayed exponential (delayed tau models).  To these it is possible to add a
burst and/or a truncation, and a constant component can be added to the two
exponential forms.

Use of parametric SFHs requires the :py:class:`prospect.sources.CSPSpecBasis` to
be used as the `sps` object

Continuity SFH
--------------
leja19, johnson21

Continuity Flex SFH
-------------------
leja19

Hybrid Continuity SFH
---------------------
suess21

Tabular SFH
-----------

Dirichlet SFH
-------------
leja17


