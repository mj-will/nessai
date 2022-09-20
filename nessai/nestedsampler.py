# -*- coding: utf-8 -*-
"""
Deprecated submodule that will be removed in a future release.
Use `nessai.samplers.nestedsampler` instead.
"""
from warnings import warn

from .samplers.nestedsampler import NestedSampler  # noqa

msg = (
    "`nessai.nestedsampler` is deprecated and will be removed in a future "
    "release. Use `nessai.samplers.nestedsampler` instead."
)
warn(msg, FutureWarning)
