# -*- coding: utf-8 -*-
"""
Tests for rescaling functions
"""
import numpy as np
import pytest
from unittest.mock import patch


from nessai.utils.rescaling import (
    configure_edge_detection,
    detect_edge,
    determine_rescaled_bounds,
    exp_with_log_jacobian,
    inverse_rescale_minus_one_to_one,
    inverse_rescale_zero_to_one,
    logistic_function,
    log_with_log_jacobian,
    rescale_minus_one_to_one,
    rescale_zero_to_one,
    logit,
    sigmoid,
)


def test_rescale_minus_one_to_one():
    """Assert rescaling is correctly applied."""
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
    expected = np.array([-1, -0.5, 0.0, 0.5, 1.0])
    x_out, log_j = rescale_minus_one_to_one(x, -5, 5)
    np.testing.assert_array_equal(x_out, expected)
    np.testing.assert_equal(log_j, np.log(2) - np.log(10))


def test_inverse_rescale_minus_one_to_one():
    """Assert rescaling is correctly applied."""
    expected = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
    x = np.array([-1, -0.5, 0.0, 0.5, 1.0])
    x_out, log_j = inverse_rescale_minus_one_to_one(x, -5, 5)
    np.testing.assert_array_equal(x_out, expected)
    np.testing.assert_equal(log_j, -np.log(2) + np.log(10))


def test_rescale_zero_to_one():
    """Assert rescaling is correctly applied."""
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
    expected = np.array([0, 0.25, 0.5, 0.75, 1.0])
    x_out, log_j = rescale_zero_to_one(x, -5, 5)
    np.testing.assert_array_equal(x_out, expected)
    np.testing.assert_equal(log_j, -np.log(10))


def test_inverse_rescale_zero_to_one():
    """Assert rescaling is correctly applied."""
    expected = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x_out, log_j = inverse_rescale_zero_to_one(x, -5, 5)
    np.testing.assert_array_equal(x_out, expected)
    np.testing.assert_equal(log_j, np.log(10))


@pytest.mark.parametrize(
    "test, expected",
    [(False, False), ("lower", False), ("upper", "upper"), ("both", "both")],
)
def test_detect_edge_test(test, expected):
    """Test detect edge test mode which should skip the main function."""
    with patch("numpy.histogram") as m:
        out = detect_edge(1, test=test, allowed_bounds=["upper"])
    assert out == expected
    m.assert_not_called()


@pytest.mark.parametrize(
    "hist_values, kwargs, expected",
    [
        ([0, 0, 10], {}, "upper"),
        ([10, 0, 0], {}, "lower"),
        ([0, 10, 0], {"allow_none": True}, False),
        ([10, 0, 10], {"allow_both": True}, "both"),
        ([0.2, 10, 0.1], {"allow_none": False}, "lower"),
        ([10, 0, 0], {"allowed_bounds": ["upper"], "allow_none": True}, False),
    ],
)
def test_detect_edge_max_location(hist_values, kwargs, expected):
    """Test detect edge based on location of the max"""
    with patch("numpy.histogram", return_value=(hist_values, [1, 2, 3])) as m:
        out = detect_edge(
            [5, 6], nbins=3, x_range=[-2, 2], percent=0.1, **kwargs
        )
    m.assert_called_once_with([5, 6], bins=3, density=True, range=[-2, 2])
    assert out == expected


def test_detect_edge_auto_bins():
    """Assert auto bins is called if nbins is auto"""
    with patch(
        "numpy.histogram", return_value=([10, 0, 0], [1, 2, 3])
    ) as m, patch("nessai.utils.rescaling.auto_bins", return_value=4) as mab:
        out = detect_edge([5, 6], nbins="auto")
    m.assert_called_once_with([5, 6], bins=4, density=True, range=None)
    mab.assert_called_once_with([5, 6])
    assert out == "lower"


def test_detect_edge_invalid_bound():
    """Assert that invalid allowed bounds raise an error"""
    with pytest.raises(RuntimeError) as excinfo:
        detect_edge(1, allowed_bounds=["both"])
    assert "Unknown allowed bounds: ['both']" in str(excinfo.value)


def test_configure_edge_detection_detect_edges():
    """Test configuring edge detection."""
    expected = {"x": 1, "allow_none": True, "cutoff": 0.5}
    out = configure_edge_detection({"x": 1}, detect_edges=True)
    assert out == expected


def test_configure_edge_detection_no_detect_edges():
    """Test configuring edge detection when set to False.

    Also test behaviour when the dictionary is None.
    """
    expected = {"allow_none": False, "cutoff": 0.0}
    out = configure_edge_detection(None, False)
    assert out == expected


@pytest.mark.parametrize(
    "prior_min, prior_max, x_min, x_max, kwargs, expected",
    [
        (-10, 8, -2, 2, {"inversion": False}, (-5, 4)),
        (-10, 8, -2, 2, {"inversion": True, "invert": False}, (-5, 4)),
        (-10, 8, -2, 2, {"inversion": False, "invert": "lower"}, (-5, 4)),
        (-10, 6, -2, 2, {"invert": "lower", "inversion": True}, (-2, 2)),
        (-10, 6, -2, 2, {"invert": "upper", "inversion": True}, (-3, 3)),
        (-10, 6, -2, 2, {"invert": "both", "inversion": True}, (-0.5, 1.5)),
    ],
)
def test_determine_rescaled_bounds(
    prior_min, prior_max, x_min, x_max, kwargs, expected
):
    """Assert the correct rescaled prior bounds are returned."""
    out = determine_rescaled_bounds(
        prior_min, prior_max, x_min, x_max, **kwargs
    )
    assert out == expected


def test_determine_rescaled_bounds_min_max_equal():
    """Assert an error is raised if the min and max are equal;"""
    with pytest.raises(ValueError) as excinfo:
        determine_rescaled_bounds(-1, 1, 0.5, 0.5)
    assert "New minimum and maximum are equal" in str(excinfo.value)


def test_determine_rescaled_bounds_invalid_invert():
    """Assert an error is raised if invert is not a valid value."""
    with pytest.raises(ValueError) as excinfo:
        determine_rescaled_bounds(
            -1, 1, -0.5, 0.5, invert="test", inversion=True
        )
    assert "Invalid value for `invert`: test" in str(excinfo.value)


@pytest.mark.parametrize(
    "x, y, log_J", [(0.0, -np.inf, np.inf), (1.0, np.inf, np.inf)]
)
def test_logit_bounds(x, y, log_J):
    """
    Test logit at the bounds
    """
    with pytest.warns(RuntimeWarning):
        assert logit(x, eps=0) == (y, log_J)


@pytest.mark.parametrize(
    "x, y, log_J", [(np.inf, 1, -np.inf), (-np.inf, 0, -np.inf)]
)
def test_sigmoid_bounds(x, y, log_J):
    """
    Test sigmoid for inf
    """
    assert sigmoid(x) == (y, log_J)


@pytest.mark.parametrize("p", [1e-5, 0.5, 1.0 - 1e-5])
@pytest.mark.parametrize("eps", [False, 1e-12])
def test_logit_sigmoid(p, eps):
    """
    Test invertibility of sigmoid(logit(x))
    """
    x = logit(p, eps=eps)
    y = sigmoid(x[0])
    np.testing.assert_almost_equal(p, y[0], decimal=10)
    np.testing.assert_almost_equal(x[1] + y[1], 0.0, decimal=10)


@pytest.mark.parametrize("p", [-10.0, -1.0, 0.0, 1.0, 10.0])
@pytest.mark.parametrize("eps", [False, 1e-12])
def test_sigmoid_logit(p, eps):
    """
    Test invertibility of logit(sigmoid(x))
    """
    x = sigmoid(p)
    y = logit(x[0], eps=eps)
    np.testing.assert_almost_equal(p, y[0], decimal=10)
    np.testing.assert_almost_equal(x[1] + y[1], 0.0, decimal=10)


def test_logistic_function():
    """Assert correct value is returned when k and x0 are specified"""
    assert logistic_function(0.0, 3.0, 2.0) == (1 / (1 + np.exp(6)))


def test_logistic_function_reference():
    """Compare the logistic function to a reference from scipy"""
    from scipy.special import expit

    x = np.array([-5.0, 0.0, 5.0])
    np.testing.assert_array_equal(logistic_function(x), expit(x))


def test_log():
    """Assert the correct values are returned"""
    x = np.random.rand(10)
    x_log, logj_log = log_with_log_jacobian(x)
    np.testing.assert_array_equal(x_log, np.log(x))
    np.testing.assert_array_equal(logj_log, -np.log(x))


def test_exp():
    """Assert the correct values are returned"""
    x = np.random.randn(10)
    x_exp, logj = exp_with_log_jacobian(x)
    np.testing.assert_array_equal(x_exp, np.exp(x))
    np.testing.assert_array_equal(logj, x)


def test_log_exp_inverse():
    """Assert the log and exp functions are the inverse of each other"""
    x = np.random.rand(10)
    x_log, logj_log = log_with_log_jacobian(x)
    x_out, logj_exp = exp_with_log_jacobian(x_log)
    np.testing.assert_almost_equal(x_out, x, decimal=14)
    np.testing.assert_almost_equal(logj_log, -logj_exp, decimal=14)
