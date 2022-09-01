# -*- coding: utf-8 -*-
"""
Tests for the stats related utilities.
"""
import numpy as np

from nessai.utils.stats import effective_sample_size, rolling_mean


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
