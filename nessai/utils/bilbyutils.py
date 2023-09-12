# -*- coding: utf-8 -*-
"""
Utilities to make interfacing with bilby easier.
"""
from warnings import warn

from .settings import get_all_kwargs, get_run_kwargs_list  # noqa

msg = (
    "`nessai.utils.bilbyutils` is deprecated and will be removed in a future "
    "release. Use `nessai.utils.settings` instead."
)
warn(msg, FutureWarning)
