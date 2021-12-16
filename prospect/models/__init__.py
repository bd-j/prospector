"""This module includes objects that store parameter specfications and
efficiently convert between parameter dictionaries and parameter vectors
necessary for fitting algorithms.  There are submodules for parameter priors,
common parameter transformations, and pre-defined sets of parameter
specifications.
"""

from .sedmodel import ProspectorParams, SedModel, SpecModel, PolySpecModel, SplineSpecModel

__all__ = ["ProspectorParams",
           "SpecModel", "PolySpecModel", "SplineSpecModel",
           "SedModel"]
