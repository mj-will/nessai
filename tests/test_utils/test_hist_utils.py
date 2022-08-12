# -*- coding: utf-8 -*-
"""
Tests for utilities related to producing histograms.
"""
import numpy as np
import pytest

from nessai.utils.hist import auto_bins


def test_auto_bins_max_bins():
    """Test the autobin function returns the max bins"""
    assert auto_bins(np.random.rand(100), max_bins=2) <= 2


def test_auto_bins_single_point():
    """Test to ensure the function produces a result with one sample"""
    assert auto_bins(np.random.rand()) >= 1


def test_auto_bins_no_samples():
    """Test to ensure the function produces a result with one sample"""
    with pytest.raises(RuntimeError) as excinfo:
        assert auto_bins([])
    assert "Input array is empty!" in str(excinfo.value)
