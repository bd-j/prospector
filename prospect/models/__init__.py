"""This module includes objects that store parameter specfications and
efficiently convert between parameter dictionaries and parameter vectors
necessary for fitting algorithms.  There are submodules for parameter priors,
common parameter transformations, and pre-defined sets of parameter
specifications.
"""


from .parameters import ProspectorParams
from .sedmodel import SpecModel, HyperSpecModel, AGNSpecModel, AGNPolySpecModel


__all__ = ["ProspectorParams",
           "SpecModel",
           "HyperSpecModel",
           "AGNSpecModel",
           "AGNPolySpecModel"
           ]

