# -*- coding: utf-8 -*-
"""
Functions related to computing the posterior distribution
"""
from .draw import draw_posterior_samples
from .weights import compute_weights

__all__ = [
    "compute_weights",
    "draw_posterior_samples",
]
