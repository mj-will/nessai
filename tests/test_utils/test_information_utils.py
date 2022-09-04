# -*- coding: utf-8 -*-
"""
Test information utilities.
"""
import numpy as np

from nessai.utils.information import differential_entropy


def test_differential_entropy():
    """Assert the correct value is returned"""
    x = np.random.randn(10)
    np.testing.assert_equal(differential_entropy(x), -np.mean(x))
