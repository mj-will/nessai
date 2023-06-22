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


def test_ess():
    """Test the effective sample size"""
    log_w = np.zeros(10)
    np.testing.assert_almost_equal(effective_sample_size(log_w), 10)
    # Make the input array remains unchanged
    assert (log_w == 0.0).all()


def test_rolling_mean():
    """Test the rolling mean."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = np.array([4.0 / 3.0, 2.0, 3.0, 4.0, 5.0, 17.0 / 3.0])
    out = rolling_mean(x, N=3)
    np.testing.assert_array_almost_equal(out, expected, decimal=15)


def test_weighted_quantile_equal_weights():
    """Test the weighted quanitle method"""
    x = [1, 2, 3, 4, 5]
    quantile = 0.5
    out = weighted_quantile(x, quantile)
    assert out == 3.0


def test_weighted_quantile_different_weights():
    """Test the weighted quantile method.

    Based on example 8 from https://aakinshin.net/posts/weighted-quantiles/
    """
    x = [1, 2, 3, 4, 5]
    w = [0.4, 0.4, 0.05, 0.05, 0.1]
    quantile = 0.5
    out = weighted_quantile(x, quantile, log_weights=np.log(w))
    np.testing.assert_almost_equal(out, 1.8416, decimal=4)


def test_weighted_quantile_value_error_quantile():
    """Assert an error is raised if the quantiles are invalid"""
    with pytest.raises(ValueError, match=r"Quantiles should be in \[0, 1\]"):
        weighted_quantile([1, 2, 3], quantiles=1.5)


def test_weighted_quantile_value_error_neff():
    """Assert an error is raised if the ESS is not finite."""
    with pytest.raises(
        ValueError, match=r"Effective sample size is not finite.*"
    ):
        weighted_quantile(
            [1, 2], 0.5, log_weights=np.array([np.NINF, np.NINF])
        )
