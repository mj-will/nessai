# -*- coding: utf-8 -*-
"""
Global configuration for nessai.
"""
import numpy as np

EPS = 1e-7
"""Epsilon value used for numerical stability"""

# Settings for live points
LOGL_DTYPE = "f8"
IT_DTYPE = "i4"
DEFAULT_FLOAT_DTYPE = "f8"
DEFAULT_FLOAT_VALUE = np.nan
CORE_PARAMETERS = ["logP", "logL", "it"]
CORE_PARAMETERS_DEFAULTS = [np.nan, np.nan, 0]
CORE_PARAMETERS_DTYPE = [DEFAULT_FLOAT_DTYPE, LOGL_DTYPE, IT_DTYPE]
EXTRA_PARAMETERS = []
EXTRA_PARAMETERS_DEFAULTS = []
EXTRA_PARAMETERS_DTYPE = []
NON_SAMPLING_PARAMETERS = CORE_PARAMETERS + EXTRA_PARAMETERS
NON_SAMPLING_DEFAULTS = CORE_PARAMETERS_DEFAULTS + EXTRA_PARAMETERS_DEFAULTS
NON_SAMPLING_DEFAULT_DTYPE = CORE_PARAMETERS_DTYPE + EXTRA_PARAMETERS_DTYPE
# Plotting config
DISABLE_STYLE = False
"""Disable nessai's custom plotting style globally.

Useful since all plotting functions use the
:py:func:`~nessai.plot.nessai_style` decorator by default.
"""
SNS_STYLE = "ticks"
"""Default seaborn style."""
BASE_COLOUR = "#02979d"
"""Base colour for plots."""
HIGHLIGHT_COLOUR = "#f5b754"
"""Highlight colour for plots."""
LINE_COLOURS = ["#4575b4", "#d73027", "#fad117", "#ff8c00"]
"""Default line colours."""
LINE_STYLES = ["-", "--", ":", "-."]
"""Default line styles."""
