# -*- coding: utf-8 -*-
"""Different nested samplers available in nessai.

All samplers inherit from :py:obj:`nessai.samplers.base.BaseNestedSampler`.
"""
from .nestedsampler import NestedSampler
from .importancesampler import ImportanceNestedSampler

__all__ = [
    "ImportanceNestedSampler",
    "NestedSampler",
]
