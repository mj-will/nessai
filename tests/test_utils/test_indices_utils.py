# -*- coding: utf-8 -*-
"""
Tests for utilies related to insertion indices.
"""
import numpy as np
import pytest

from nessai.utils.indices import compute_indices_ks_test


@pytest.mark.parametrize("mode", ['D+', 'D-'])
def test_ks_test(mode):
    """
    Test KS test for insertion indices with a specifed mode
    """
    indices = np.random.randint(0, 1000, 1000)
    out = compute_indices_ks_test(indices, 1000, mode=mode)
    assert all([o > 0. for o in out])


def test_ks_test_undefined_mode():
    """
    Test KS test for insertion indices with undefined mode
    """
    indices = np.random.randint(0, 1000, 1000)
    with pytest.raises(RuntimeError):
        compute_indices_ks_test(indices, 1000, mode='two-sided')


def test_ks_test_empty_indices():
    """
    Test KS test for insertion indices with empty input array
    """
    out = compute_indices_ks_test([], 1000, mode='D+')
    assert all(o is None for o in out)
