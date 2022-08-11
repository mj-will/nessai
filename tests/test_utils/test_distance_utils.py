# -*- coding: utf-8 -*-
"""
Test utilities for computing distances.
"""
import numpy as np
import pytest
from unittest.mock import patch

from nessai.utils.distance import compute_minimum_distances


def test_minimum_distance():
    """Test the minimum distance function"""
    samples = np.array([[1], [2], [4]])
    d = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    with patch("nessai.utils.distance.distance.cdist", return_value=d) as mock:
        dmin = compute_minimum_distances(samples, metric="test")

    mock.assert_called_once_with(samples, samples, "test")
    np.testing.assert_array_equal(dmin, np.array([1, 1, 2]))


@pytest.mark.integration_test
@pytest.mark.parametrize("metric", ["euclidean", "minkowski"])
def test_minimum_distance_integration(metric):
    """Integration test for the minimum distance"""
    samples = np.array([[1], [2], [4]])
    dmin = compute_minimum_distances(samples, metric=metric)
    np.testing.assert_array_equal(dmin, np.array([1, 1, 2]))
