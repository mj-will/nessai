# -*- coding: utf-8 -*-
"""
Global configuration for nessai.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class LivepointsConfig:
    """Configuration for live points."""

    logl_dtype: str = "f8"
    """Default log-likelihood dtype"""
    it_dtype: str = "i4"
    """Default dtype for iteration parameter"""
    it_default: int = 0
    """Default value for the iteration parameter"""
    default_float_dtype: str = "f8"
    """Default dtype for parameters"""
    default_float_value: float = np.nan
    """Default value for parameters"""
    core_parameters: List[str] = field(
        default_factory=lambda: ["logP", "logL", "it"]
    )
    """List of the core non-sampling parameters included in all live points"""
    extra_parameters: List[str] = field(default_factory=lambda: [])
    """Additional extra parameters included in live points"""
    extra_parameters_dtype: List[str] = field(default_factory=lambda: [])
    """Defaults dtype for extra parameters"""
    extra_parameters_defaults: Tuple = field(default_factory=lambda: ())
    """Default values for additional extra extra"""

    _core_parameter_dtype: List[str] = None
    _core_parameter_defaults: Tuple = None
    _non_sampling_defaults: List = None
    _non_sampling_parameters: Tuple = None
    _non_sampling_dtype: List[str] = None

    @property
    def core_parameters_dtype(self) -> List[str]:
        """List of dtypes for the core non-sampling parameters"""
        if self._core_parameter_dtype is None:
            self._core_parameter_dtype = [
                self.default_float_dtype,
                self.logl_dtype,
                self.it_dtype,
            ]
        return self._core_parameter_dtype

    @property
    def core_parameters_defaults(self) -> Tuple:
        """Tuple of default values for core non-sampling parameters."""
        if self._core_parameter_defaults is None:
            self._core_parameter_defaults = (
                self.default_float_value,
                self.default_float_value,
                self.it_default,
            )
        return self._core_parameter_defaults

    @property
    def non_sampling_parameters(self) -> List[str]:
        """List of all the non-sampling parameters"""
        if self._non_sampling_parameters is None:
            self._non_sampling_parameters = (
                self.core_parameters + self.extra_parameters
            )
        return self._non_sampling_parameters

    @property
    def non_sampling_defaults(self) -> Tuple:
        """List of default values for all the non-sampling parameters"""
        if self._non_sampling_defaults is None:
            self._non_sampling_defaults = (
                self.core_parameters_defaults + self.extra_parameters_defaults
            )
        return self._non_sampling_defaults

    @property
    def non_sampling_dtype(self) -> List[str]:
        """List of the dtype for all the non-sampling parameters"""
        if self._non_sampling_dtype is None:
            self._non_sampling_dtype = (
                self.core_parameters_dtype + self.extra_parameters_dtype
            )
        return self._non_sampling_dtype

    def reset(self) -> None:
        """Reset the extra parameters and properties"""
        self.extra_parameters = []
        self.extra_parameters_defaults = ()
        self.extra_parameters_dtype = []
        self.reset_properties()

    def reset_properties(self) -> None:
        """Reset the cached properties"""
        self._core_parameter_dtype = None
        self._core_parameter_defaults = None
        self._non_sampling_defaults = None
        self._non_sampling_parameters = None
        self._non_sampling_dtype = None


@dataclass
class PlottingConfig:
    """Configuration for plotting."""

    disable_style: bool = False
    """Disable nessai's custom plotting style globally.

    Useful since all plotting functions use the
    :py:func:`~nessai.plot.nessai_style` decorator by default.
    """
    sns_style: str = "ticks"
    """Default seaborn style."""
    base_colour: str = "#02979d"
    """Base colour for plots."""
    highlight_colour: str = "#f5b754"
    """Highlight colour for plots."""
    line_colours: List[str] = field(
        default_factory=lambda: ["#4575b4", "#d73027", "#fad117", "#ff8c00"]
    )
    """Default line colours."""
    line_styles: List[str] = field(
        default_factory=lambda: ["-", "--", ":", "-."]
    )
    """Default line styles."""


livepoints = LivepointsConfig()
plotting = PlottingConfig()
