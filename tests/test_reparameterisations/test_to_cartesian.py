# -*- coding: utf-8 -*-
"""
Test the ToCartesian reparameterisation.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.reparameterisations import ToCartesian
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture
def reparam():
    return create_autospec(ToCartesian)


@pytest.fixture
def x():
    return numpy_array_to_live_points(np.array([[0.0], [1.0], [2.0]]), ["x"])


@pytest.mark.parametrize("mode", ["duplicate", "half", "split"])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_init(reparam, mode, scale):
    prior_bounds = {"x": [0.0, 1.0]}
    """Test the init method"""

    def side_effect(scale, prior_bounds=None):
        reparam.scale = scale
        reparam.parameters = ["x", "y"]
        reparam.prior_bounds = prior_bounds

    with patch(
        "nessai.reparameterisations.Angle.__init__", side_effect=side_effect
    ) as super_init:
        ToCartesian.__init__(
            reparam, mode=mode, scale=scale, prior_bounds=prior_bounds
        )
    assert reparam.mode == mode
    assert reparam._zero_bound is False
    assert reparam._k == 1.0
    super_init.assert_called_once_with(scale=scale, prior_bounds=prior_bounds)


def test_init_invalid_mode(reparam):
    """Test the init method with an invalid mode"""
    with patch("nessai.reparameterisations.Angle.__init__"), pytest.raises(
        RuntimeError
    ) as excinfo:
        ToCartesian.__init__(reparam, mode="double")
    assert "Unknown mode" in str(excinfo.value)


def test_rescale_angle_split(reparam, x):
    """Test method for rescaling the 'angle'.

    For ToCartesian this is the rescaling applied to the parameter.
    """
    reparam.parameters = ["x"]
    reparam.prior_bounds = {"x": [0, 2]}
    reparam.mode = "split"
    reparam.scale = np.pi
    expected_angle = np.array([0.0, -np.pi / 2, np.pi])

    x_prime = numpy_array_to_live_points(np.zeros([3, 1]), ["x_prime"])
    log_j = np.zeros(3)

    with patch("numpy.random.choice", return_value=np.array([1])):
        angle, x_out, x_prime_out, log_j_out = ToCartesian._rescale_angle(
            reparam, x, x_prime, log_j, compute_radius=False
        )

    np.testing.assert_array_equal(angle, expected_angle)
    assert_structured_arrays_equal(x_out, x)
    assert_structured_arrays_equal(x_prime_out, x_prime)
    np.testing.assert_equal(log_j_out, -np.log(2))


@pytest.mark.parametrize(
    "args",
    [
        {"mode": "split", "compute_radius": True},
        {"mode": "duplicate", "compute_radius": False},
    ],
)
def test_rescale_angle_duplicate_or_compute_radius(reparam, x, args):
    """Test method for rescaling the 'angle'.

    For ToCartesian this is the rescaling applied to the parameter.
    """
    reparam.parameters = ["x"]
    reparam.prior_bounds = {"x": [0, 2]}
    reparam.mode = args["mode"]
    reparam.scale = np.pi

    expected_angle = np.array([0.0, np.pi / 2, np.pi, 0.0, -np.pi / 2, -np.pi])

    x_prime = numpy_array_to_live_points(np.zeros([3, 1]), ["x_prime"])
    log_j = np.zeros(3)

    with patch("numpy.random.choice", return_value=np.array([1])):
        angle, x_out, x_prime_out, log_j_out = ToCartesian._rescale_angle(
            reparam, x, x_prime, log_j, compute_radius=args["compute_radius"]
        )

    np.testing.assert_array_equal(angle, expected_angle)
    assert_structured_arrays_equal(x_out, np.concatenate([x, x]))
    assert_structured_arrays_equal(
        x_prime_out, np.concatenate([x_prime, x_prime])
    )
    np.testing.assert_equal(log_j_out, -np.log(2))


def test_inverse_rescale_angle(reparam):
    """Test the inverse method for rescaling the 'angle'."""
    reparam.parameters = ["x"]
    reparam.prior_bounds = {"x": [0, 2]}

    x = numpy_array_to_live_points(np.array([[0.0], [-0.5], [1.0]]), ["x"])
    expected_x = x.copy()
    expected_x["x"] = np.array([0.0, 1.0, 2.0])

    x_prime = numpy_array_to_live_points(
        np.array([0.0, 1.0, 3.0]), ["x_prime"]
    )
    log_j = np.zeros(3)

    x_out, x_prime_out, log_j_out = ToCartesian._inverse_rescale_angle(
        reparam, x, x_prime, log_j
    )

    assert_structured_arrays_equal(x_out, expected_x)
    assert_structured_arrays_equal(x_prime_out, x_prime)
    np.testing.assert_equal(log_j_out, np.log(2))
