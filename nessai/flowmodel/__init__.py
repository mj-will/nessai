# -*- coding: utf-8 -*-
"""
Classes for interfacing with flows. This includes training and sampling.
"""
from .base import FlowModel
from .utils import update_config

__all__ = [
    "FlowModel",
    "update_config",
]
