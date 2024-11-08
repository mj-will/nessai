# -*- coding: utf-8 -*-
"""
Functions and objects related to reparametersiations for use in the
``reparameterisations`` dictionary.

See the documentation for an in-depth description of how to use these
functions and classes.
"""

import logging

from .angle import Angle, AnglePair, ToCartesian
from .base import Reparameterisation
from .combined import CombinedReparameterisation
from .discrete import Dequantise
from .null import NullReparameterisation
from .rescale import Rescale, RescaleToBounds, ScaleAndShift
from .utils import (
    KnownReparameterisation,
    ReparameterisationDict,
    get_reparameterisation,
)

logger = logging.getLogger(__name__)


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
    "KnownReparameterisation",
    "NullReparameterisation",
    "Reparameterisation",
    "Rescale",
    "RescaleToBounds",
    "ToCartesian",
    "get_reparameterisation",
]
