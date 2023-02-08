"""Tests for the config module"""
from nessai.config import LivepointsConfig
import numpy as np


def test_livepoint_config_reset_properties():
    """Assert the properties are reset"""
    conf = LivepointsConfig()

    assert conf.core_parameters == ["logP", "logL", "it"]
    assert conf.core_parameters_dtype == ["f8", "f8", "i4"]
    assert conf.core_parameters_defaults == (np.nan, np.nan, 0)
    assert conf.extra_parameters == []
    assert conf.extra_parameters_dtype == []
    assert conf.extra_parameters_defaults == ()

    assert conf.non_sampling_dtype == ["f8", "f8", "i4"]
    assert conf.non_sampling_defaults == (np.nan, np.nan, 0)

    conf.default_float_value = -np.inf
    conf.extra_parameters = ["a"]
    conf.extra_parameters_defaults = (0.0,)
    conf.extra_parameters_dtype = ["f4"]

    assert conf.non_sampling_defaults == (np.nan, np.nan, 0)

    conf.reset_properties()

    assert conf.core_parameters_defaults == (-np.inf, -np.inf, 0)
    assert conf.non_sampling_parameters == ["logP", "logL", "it", "a"]
    assert conf.non_sampling_dtype == ["f8", "f8", "i4", "f4"]
    assert conf.non_sampling_defaults == (-np.inf, -np.inf, 0, 0.0)


def test_livepoint_config_reset():
    """Assert the reset method clears the values"""
    conf = LivepointsConfig(
        extra_parameters=["a", "b"],
        extra_parameters_dtype=["f4", "i4"],
        extra_parameters_defaults=[0.0, 0],
    )
    assert conf.non_sampling_parameters == ["logP", "logL", "it", "a", "b"]
    conf.reset()
    assert conf.extra_parameters == []
    assert conf.extra_parameters_dtype == []
    assert conf.extra_parameters_defaults == ()
    assert conf.non_sampling_parameters == ["logP", "logL", "it"]
