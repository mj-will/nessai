# -*- coding: utf-8 -*-
"""
Code related to the implementation of normalising flows and some complete
implementations.
"""
from .base import BaseFlow, NFlow
from .maf import MaskedAutoregressiveFlow
from .nsf import NeuralSplineFlow
from .realnvp import RealNVP
from .utils import (
    add_noise_to_parameters,
    configure_model,
    get_n_neurons,
    set_affine_parameters,
    reset_weights,
    reset_permutations,
)


__all__ = ["BaseFlow",
           "NFlow",
           "MaskedAutoregressiveFlow",
           "NeuralSplineFlow",
           "RealNVP",
           "add_noise_to_parameters",
           "configure_model",
           "get_n_neurons",
           "set_affine_parameters",
           "reset_weights",
           "reset_permutations"]
