# -*- coding: utf-8 -*-
"""
Tests for the stats related utilities.
"""
import numpy as np
import pytest

from nessai.utils.stats import (
    effective_sample_size,
    rolling_mean,
    weighted_quantile,
)


def test_rolling_mean():
    """Test the rolling mean."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = np.array([4.0 / 3.0, 2.0, 3.0, 4.0, 5.0, 17.0 / 3.0])
    out = rolling_mean(x, N=3)
    np.testing.assert_array_almost_equal(out, expected, decimal=15)


def test_effective_sample_size():
    """Test the effective samples size"""
    log_w = np.log(np.random.rand(100))
    ess = effective_sample_size(log_w)
    assert ess > 0


def test_weighted_quantile():
    """Test the weighted quantile"""
    samples = np.random.randn(100)
    weights = np.random.randn(100)
    out = weighted_quantile(samples, 0.5, weights=weights)
    assert samples.min() < out < samples.max()


def test_weighted_quantile_invalid_quantile():
    """Assert an error is raised if the quantile is not in [0, 1]"""
    with pytest.raises(ValueError) as excinfo:
        weighted_quantile(np.random.rand(10), -0.5)
    assert 'Quantiles should be in [0, 1]' in str(excinfo.value)
