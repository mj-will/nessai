# -*- coding: utf-8 -*-
"""
Functions and objects related to reparametersiations for use in the
``reparameterisations`` dictionary.

See the documentation for an in-depth description of how to use these
functions and classes.
"""
from .angle import Angle, AnglePair, ToCartesian
from .base import Reparameterisation
from .combined import CombinedReparameterisation
from .discrete import Dequantise
from .null import NullReparameterisation
from .rescale import Rescale, RescaleToBounds, ScaleAndShift
from .utils import get_reparameterisation


default_reparameterisations = {
    "default": (RescaleToBounds, None),
    "rescaletobounds": (RescaleToBounds, None),
    "rescale-to-bounds": (RescaleToBounds, None),
    "offset": (RescaleToBounds, {"offset": True}),
    "inversion": (
        RescaleToBounds,
        {
            "detect_edges": True,
            "boundary_inversion": True,
            "inversion_type": "split",
        },
    ),
    "inversion-duplicate": (
        RescaleToBounds,
        {
            "detect_edges": True,
            "boundary_inversion": True,
            "inversion_type": "duplicate",
        },
    ),
    "logit": (
        RescaleToBounds,
        {
            "rescale_bounds": [0.0, 1.0],
            "update_bounds": False,
            "post_rescaling": "logit",
        },
    ),
    "log-rescale": (
        RescaleToBounds,
        {
            "rescale_bounds": [0.0, 1.0],
            "update_bounds": False,
            "post_rescaling": "log",
        },
    ),
    "scale": (Rescale, None),
    "scaleandshift": (ScaleAndShift, None),
    "rescale": (Rescale, None),
    "zscore": (
        ScaleAndShift,
        {"estimate_scale": True, "estimate_shift": True},
    ),
    "z-score": (
        ScaleAndShift,
        {"estimate_scale": True, "estimate_shift": True},
    ),
    "zscore-gaussian-cdf": (
        ScaleAndShift,
        {
            "estimate_scale": True,
            "estimate_shift": True,
            "post_rescaling": "gaussian_cdf",
        },
    ),
    "z-score-gaussian-cdf": (
        ScaleAndShift,
        {
            "estimate_scale": True,
            "estimate_shift": True,
            "post_rescaling": "gaussian_cdf",
        },
    ),
    "z-score-logit": (
        ScaleAndShift,
        {
            "estimate_scale": True,
            "estimate_shift": True,
            "pre_rescaling": "logit",
        },
    ),
    "zscore-logit": (
        ScaleAndShift,
        {
            "estimate_scale": True,
            "estimate_shift": True,
            "pre_rescaling": "logit",
        },
    ),
    "z-score-inv-gaussian-cdf": (
        ScaleAndShift,
        {
            "estimate_scale": True,
            "estimate_shift": True,
            "pre_rescaling": "inv_gaussian_cdf",
        },
    ),
    "zscore-inv-gaussian-cdf": (
        ScaleAndShift,
        {
            "estimate_scale": True,
            "estimate_shift": True,
            "pre_rescaling": "inv_gaussian_cdf",
        },
    ),
    "angle": (Angle, {}),
    "angle-pi": (Angle, {"scale": 2.0, "prior": "uniform"}),
    "angle-2pi": (Angle, {"scale": 1.0, "prior": "uniform"}),
    "angle-sine": (RescaleToBounds, None),
    "angle-cosine": (RescaleToBounds, None),
    "angle-pair": (AnglePair, None),
    "periodic": (Angle, {"scale": None}),
    "to-cartesian": (ToCartesian, None),
    "dequantise": (Dequantise, None),
    "dequantise-logit": (
        Dequantise,
        {
            "rescale_bounds": [0.0, 1.0],
            "update_bounds": False,
            "post_rescaling": "logit",
        },
    ),
    "none": (NullReparameterisation, None),
    "null": (NullReparameterisation, None),
    None: (NullReparameterisation, None),
}


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
