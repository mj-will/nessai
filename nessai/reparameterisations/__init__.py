# -*- coding: utf-8 -*-
"""
Functions and objects related to reparametersiations for use in the
``reparameterisations`` dictionary.

See the documentation for an in-depth description of how to use these
functions and classes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..utils.entry_points import get_entry_points
from .angle import Angle, AnglePair, ToCartesian
from .base import Reparameterisation
from .combined import CombinedReparameterisation
from .discrete import Dequantise
from .null import NullReparameterisation
from .rescale import Rescale, RescaleToBounds, ScaleAndShift
from .utils import get_reparameterisation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnownReparameterisation:
    """Dataclass to store the reparameterisation class and keyword arguments"""

    name: str
    class_fn: Reparameterisation
    keyword_arguments: dict[str:Any] = field(default_factory=dict)


class ReparameterisationDict(dict):
    """Dictionary of reparameterisations

    This dictionary is used to store the known reparameterisations and
    provides a method to add new reparameterisations.
    """

    def add_reparameterisation(self, name, class_fn, keyword_arguments=None):
        """Add a new reparameterisation to the dictionary

        Parameters
        ----------
        name : str
            Name of the reparameterisation.
        class_fn : Reparameterisation
            Reparameterisation class.
        keyword_arguments : dict, optional
            Keyword arguments for the reparameterisation.
        """
        if keyword_arguments is None:
            keyword_arguments = {}
        if name in self:
            raise ValueError(f"Reparameterisation {name} already exists")
        self[name] = KnownReparameterisation(name, class_fn, keyword_arguments)

    def add_external_reparameterisations(self, group):
        entry_points = get_entry_points(group)
        for ep in entry_points.values():
            reparam = ep.load()
            if reparam is not isinstance(KnownReparameterisation):
                raise RuntimeError(
                    f"Invalid external reparameterisation: {reparam}"
                )
            elif reparam.name in self:
                raise ValueError(
                    f"Reparameterisation {reparam.name} already exists"
                )
            logger.debug(f"Adding external reparameterisation: {reparam}")
            self[reparam.name] = reparam


default_reparameterisations = ReparameterisationDict()
default_reparameterisations.add_reparameterisation("default", RescaleToBounds)
default_reparameterisations.add_reparameterisation(
    "rescaletobounds", RescaleToBounds
)
default_reparameterisations.add_reparameterisation(
    "rescale-to-bounds", RescaleToBounds
)

default_reparameterisations.add_reparameterisation(
    "offset",
    RescaleToBounds,
    {"offset": True},
)
default_reparameterisations.add_reparameterisation(
    "inversion",
    RescaleToBounds,
    {
        "detect_edges": True,
        "boundary_inversion": True,
        "inversion_type": "split",
    },
)
default_reparameterisations.add_reparameterisation(
    "inversion-duplicate",
    RescaleToBounds,
    {
        "detect_edges": True,
        "boundary_inversion": True,
        "inversion_type": "duplicate",
    },
)
default_reparameterisations.add_reparameterisation(
    "logit",
    RescaleToBounds,
    {
        "rescale_bounds": [0.0, 1.0],
        "update_bounds": False,
        "post_rescaling": "logit",
    },
)
default_reparameterisations.add_reparameterisation(
    "log-rescale",
    RescaleToBounds,
    {
        "rescale_bounds": [0.0, 1.0],
        "update_bounds": False,
        "post_rescaling": "log",
    },
)
default_reparameterisations.add_reparameterisation("scale", Rescale)
default_reparameterisations.add_reparameterisation(
    "scaleandshift", ScaleAndShift
)
default_reparameterisations.add_reparameterisation("rescale", Rescale)
default_reparameterisations.add_reparameterisation(
    "zscore",
    ScaleAndShift,
    {"estimate_scale": True, "estimate_shift": True},
)
default_reparameterisations.add_reparameterisation(
    "z-score",
    ScaleAndShift,
    {"estimate_scale": True, "estimate_shift": True},
)
default_reparameterisations.add_reparameterisation(
    "zscore-gaussian-cdf",
    ScaleAndShift,
    {
        "estimate_scale": True,
        "estimate_shift": True,
        "post_rescaling": "gaussian_cdf",
    },
)
default_reparameterisations.add_reparameterisation(
    "z-score-gaussian-cdf",
    ScaleAndShift,
    {
        "estimate_scale": True,
        "estimate_shift": True,
        "post_rescaling": "gaussian_cdf",
    },
)
default_reparameterisations.add_reparameterisation(
    "z-score-logit",
    ScaleAndShift,
    {
        "estimate_scale": True,
        "estimate_shift": True,
        "pre_rescaling": "logit",
    },
)
default_reparameterisations.add_reparameterisation(
    "zscore-logit",
    ScaleAndShift,
    {
        "estimate_scale": True,
        "estimate_shift": True,
        "pre_rescaling": "logit",
    },
)
default_reparameterisations.add_reparameterisation(
    "z-score-inv-gaussian-cdf",
    ScaleAndShift,
    {
        "estimate_scale": True,
        "estimate_shift": True,
        "pre_rescaling": "inv_gaussian_cdf",
    },
)
default_reparameterisations.add_reparameterisation(
    "zscore-inv-gaussian-cdf",
    ScaleAndShift,
    {
        "estimate_scale": True,
        "estimate_shift": True,
        "pre_rescaling": "inv_gaussian_cdf",
    },
)
default_reparameterisations.add_reparameterisation("angle", Angle, {})
default_reparameterisations.add_reparameterisation(
    "angle-pi", Angle, {"scale": 2.0, "prior": "uniform"}
)
default_reparameterisations.add_reparameterisation(
    "angle-2pi", Angle, {"scale": 1.0, "prior": "uniform"}
)
default_reparameterisations.add_reparameterisation(
    "angle-sine", RescaleToBounds
)
default_reparameterisations.add_reparameterisation(
    "angle-cosine", RescaleToBounds
)
default_reparameterisations.add_reparameterisation("angle-pair", AnglePair)
default_reparameterisations.add_reparameterisation(
    "periodic", Angle, {"scale": None}
)
default_reparameterisations.add_reparameterisation("to-cartesian", ToCartesian)
default_reparameterisations.add_reparameterisation("dequantise", Dequantise)
default_reparameterisations.add_reparameterisation(
    "dequantise-logit",
    Dequantise,
    {
        "rescale_bounds": [0.0, 1.0],
        "update_bounds": False,
        "post_rescaling": "logit",
    },
)
default_reparameterisations.add_reparameterisation(
    "none", NullReparameterisation
)
default_reparameterisations.add_reparameterisation(
    "null", NullReparameterisation
)
default_reparameterisations.add_reparameterisation(
    None, NullReparameterisation
)


default_reparameterisations.add_external_reparameterisations(
    "nessai.reparameterisations"
)


__all__ = [
    "Angle",
    "AnglePair",
    "CombinedReparameterisation",
    "Dequantise",
    "NullReparameterisation",
    "Reparameterisation",
    "Rescale",
    "RescaleToBounds",
    "ToCartesian",
    "get_reparameterisation",
]
