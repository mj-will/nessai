# -*- coding: utf-8 -*-
"""
Tests for rescaling functions
"""
import numpy as np
import pytest


from nessai.utils.rescaling import (
    logit,
    sigmoid
)


@pytest.mark.parametrize(
    "x, y, log_J",
    [(0., -np.inf, np.inf), (1., np.inf, np.inf)]
)
def test_logit_bounds(x, y, log_J):
    """
    Test logit at the bounds
    """
    with pytest.warns(RuntimeWarning):
        assert logit(x, fuzz=0) == (y, log_J)


@pytest.mark.parametrize(
    "x, y, log_J",
    [(np.inf, 1, -np.inf), (-np.inf, 0, -np.inf)]
)
def test_sigmoid_bounds(x, y, log_J):
    """
    Test sigmoid for inf
    """
    assert sigmoid(x, fuzz=0) == (y, log_J)


@pytest.mark.parametrize("p", [1e-5, 0.5, 1 - 1e-5])
def test_logit_sigmoid(p):
    """
    Test invertibility of sigmoid(logit(x))
    """
    x = logit(p, fuzz=0)
    y = sigmoid(x[0], fuzz=0)
    np.testing.assert_equal(p, y[0])
    np.testing.assert_almost_equal(x[1] + y[1], 0)


@pytest.mark.parametrize("p", [-10, -1, 0, 1, 10])
def test_sigmoid_logit(p):
    """
    Test invertibility of logit(sigmoid(x))
    """
    x = sigmoid(p, fuzz=0)
    y = logit(x[0], fuzz=0)
    np.testing.assert_almost_equal(p, y[0])
    np.testing.assert_almost_equal(x[1] + y[1], 0)
