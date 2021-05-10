# -*- coding: utf-8 -*-
"""
Tests related to nested sampling when using importance sampling.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec, MagicMock

from nessai.nestedsampler import NestedSampler
from nessai.livepoint import numpy_array_to_live_points


@pytest.fixture
def sampler():
    return create_autospec(NestedSampler)


def test_log_w_norm_decrese_equal_weights(sampler):
    """Test to make sure the value of log_w_norm descreases correctly.

    If the weights are equal, after removing a point the new norm should
    be log(n-1). This test interatively remove points and checks the value
    decreases as expected.
    """
    sampler._rejection_sampling = False
    n = 1000
    x = numpy_array_to_live_points(np.zeros([n, 1]), names=['a'])
    sampler.live_points = x
    sampler.log_w_norm = np.logaddexp.reduce(x['logW'])
    sampler.state = MagicMock()
    np.testing.assert_allclose(sampler.log_w_norm, np.log(n))
    # Can't test last point because of inf relative diff.
    for i, s in enumerate(x[:-2], start=1):
        NestedSampler._increment(sampler, s)
        np.testing.assert_allclose(sampler.log_w_norm, np.log(n-i))


def test_log_w_norm_decrese_diff_weights(sampler):
    """Test to make sure the value of log_w_norm descreases correctly.

    """
    sampler._rejection_sampling = False
    n = 1000
    x = numpy_array_to_live_points(np.zeros([n, 1]), names=['a'])
    x['logW'] = np.log(np.random.rand(n) * 1e-10)
    sampler.live_points = x
    sampler.log_w_norm = np.logaddexp.reduce(x['logW'])
    sampler.state = MagicMock()
    for i, s in enumerate(x[:-2], start=1):
        NestedSampler._increment(sampler, s)
        np.testing.assert_allclose(sampler.log_w_norm,
                                   np.logaddexp.reduce(x['logW'][i:]))


def test_log_w_norm_increase_equal_weights(sampler):
    """Test to make sure log_w_norm increases correctly"""
    sampler._rejection_sampling = False
    n = 1000
    x = numpy_array_to_live_points(np.zeros([n, 1]), names=['a'])
    y = x.copy()
    x['logL'] = np.arange(n)
    sampler.live_points = x
    sampler.log_w_norm = np.logaddexp.reduce(x['logW'])
    sampler.state = MagicMock()
    np.testing.assert_allclose(sampler.log_w_norm, np.log(n))
    y['logL'] = np.arange(n, 2 * n, ) + 1
    for i, s in enumerate(y[:-2], start=1):
        NestedSampler.insert_live_point(sampler, s)
        np.testing.assert_allclose(sampler.log_w_norm, np.log(n+i))


def test_log_w_norm_increase_diff_weights(sampler):
    """Test to make sure log_w_norm increases correctly"""
    sampler._rejection_sampling = False
    n = 1000
    x = numpy_array_to_live_points(np.zeros([n, 1]), names=['a'])
    x['logW'] = np.log(np.random.rand(n) * 1e-12)
    y = x.copy()
    x['logL'] = np.arange(n)
    sampler.live_points = x
    sampler.log_w_norm = np.logaddexp.reduce(x['logW'])
    sampler.state = MagicMock()
    y['logL'] = np.arange(n, 2 * n, ) + 1
    for i, s in enumerate(y, start=1):
        NestedSampler.insert_live_point(sampler, s)
        np.testing.assert_allclose(
            sampler.log_w_norm,
            np.logaddexp.reduce(np.concatenate([x['logW'], y['logW'][:i]])))
