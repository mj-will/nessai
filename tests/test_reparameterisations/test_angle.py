# -*- coding: utf-8 -*-
"""
Test the Angle reparameterisation
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec

from nessai.reparameterisations import Angle
from nessai.livepoint import (
    empty_structured_array,
    numpy_array_to_live_points,
    parameters_to_live_point,
)
from nessai.utils.testing import assert_structured_arrays_equal

scales = [1.0, 2.0]


@pytest.fixture(params=scales, scope="function")
def scale(request):
    return request.param


@pytest.fixture
def reparam():
    return create_autospec(Angle)


@pytest.fixture(scope="function")
def assert_invertibility(model, n=100):
    def test_invertibility(reparam, angle, radial=None):

        x = empty_structured_array(n, names=reparam.parameters)
        x_prime = empty_structured_array(n, names=reparam.prime_parameters)
        log_j = 0

        x[reparam.angle] = angle
        if radial is not None:
            x[reparam.radial] = radial

        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

        np.testing.assert_array_equal(x[reparam.angle], x_re[reparam.angle])
        if radial is not None:
            np.testing.assert_array_equal(
                x[reparam.radial], x_re[reparam.radial]
            )

        x_in = empty_structured_array(n, names=reparam.parameters)

        x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
            x_in, x_prime_re, log_j
        )

        np.testing.assert_array_almost_equal(
            x[reparam.angle], x_inv[reparam.angle]
        )
        if radial is not None:
            np.testing.assert_array_almost_equal(
                x[reparam.radial], x_inv[reparam.radial]
            )

        assert_structured_arrays_equal(x_prime_re, x_prime_inv)
        np.testing.assert_array_almost_equal(log_j_re, -log_j_inv)

        return True

    return test_invertibility


@pytest.mark.parametrize(
    "bounds, scale, expected_scale",
    [
        ([0, 2 * np.pi], None, 1.0),
        ([-1, 1], None, np.pi),
        ([0, np.pi], 1.0, 1.0),
    ],
)
def test_angle_parameter(bounds, scale, expected_scale):
    """Test init with just an angle parameter"""
    parameter = "theta"
    prior_bounds = {parameter: bounds}
    reparam = Angle(
        parameters=parameter, prior_bounds=prior_bounds, scale=scale
    )

    assert reparam.chi is not False
    assert hasattr(reparam.chi, "rvs")
    if bounds[0] == 0.0:
        assert reparam._zero_bound is True
    else:
        assert reparam._zero_bound is False
    assert reparam.has_prime_prior is False

    assert reparam.angle == parameter
    assert reparam.radial == (parameter + "_radial")
    assert reparam.radius == (parameter + "_radial")
    assert reparam.x == (parameter + "_x")
    assert reparam.y == (parameter + "_y")
    assert reparam.scale == expected_scale


def test_angle_too_many_parameters(reparam):
    """Assert an error is raised if too many parameters are given."""
    parameters = ["x", "y", "z"]
    prior_bounds = {p: [-1, 1] for p in parameters}
    with pytest.raises(RuntimeError) as excinfo:
        Angle.__init__(
            reparam, parameters=parameters, prior_bounds=prior_bounds
        )
    assert reparam.parameters == parameters
    assert "Too many parameters for Angle" in str(excinfo.value)


def test_angle_prior_uniform():
    """Assert the prior is correctly assigned for uniform prior"""
    from nessai.priors import log_2d_cartesian_prior

    reparam = Angle(
        parameters="theta",
        prior="uniform",
        prior_bounds={"theta": [0, np.pi]},
        scale=2.0,
    )
    assert reparam.prior == "uniform"
    assert reparam.has_prime_prior is True
    assert reparam._k == (2 * np.pi)
    assert reparam._prime_prior is log_2d_cartesian_prior


def test_angle_prior_sine():
    """Assert the prior is correctly assigned for sine prior"""
    from nessai.priors import log_2d_cartesian_prior_sine

    reparam = Angle(
        parameters="theta",
        prior="sine",
        prior_bounds={"theta": [0, np.pi]},
    )
    assert reparam.prior == "sine"
    assert reparam.has_prime_prior is True
    assert reparam._k == np.pi
    assert reparam._prime_prior is log_2d_cartesian_prior_sine


def test_log_prior(reparam):
    """Assert the log-prior calls the correct function"""
    reparam.chi = MagicMock()
    reparam.chi.logpdf = MagicMock()
    reparam.parameters = ["theta", "theta_radial"]
    x = {"theta": [1.0], "theta_radial": [0.5]}
    Angle.log_prior(reparam, x)
    reparam.chi.logpdf.assert_called_once_with([0.5])


def test_x_prime_log_prior(reparam):
    """ "Assert the underlying functions is called correctly"""
    x_prime = numpy_array_to_live_points(np.array([[1.0, -1.0]]), ["x", "y"])

    reparam._k = 0.5
    reparam._prime_prior = MagicMock(return_value=0.5)
    reparam.has_prior_prior = True
    reparam.prime_parameters = ["x", "y"]

    out = Angle.x_prime_log_prior(reparam, x_prime)

    reparam._prime_prior.assert_called_once_with(
        x_prime["x"],
        x_prime["y"],
        k=0.5,
    )
    assert out == 0.5


def test_x_prime_log_prior_error(reparam):
    """
    Assert an error is raised when called the prime prior if is not enabled.
    """
    reparam.has_prime_prior = False
    x = {"theta": [1.0], "theta_radial": [0.5]}
    with pytest.raises(RuntimeError) as excinfo:
        Angle.x_prime_log_prior(reparam, x)
    assert "Prime prior" in str(excinfo.value)


def test_both_parameters():
    """Test init with just an angle and radial parameter"""
    parameters = ["theta", "r"]
    prior_bounds = {
        parameters[0]: np.array([0, 2 * np.pi]),
        parameters[1]: np.array([0, 5]),
    }

    reparam = Angle(parameters=parameters, prior_bounds=prior_bounds)

    assert reparam.chi is False
    assert reparam._zero_bound is True
    assert reparam.has_prime_prior is False

    assert reparam.angle == parameters[0]
    assert reparam.radial == parameters[1]


@pytest.mark.parametrize(
    "angle_prior", [np.array([0, 2 * np.pi]), np.array([-np.pi, np.pi])]
)
def test_invertiblity_single_parameter(
    angle_prior, scale, assert_invertibility
):
    """Test the inverbility when using just an angle"""
    n = 100
    parameter = "theta"
    prior_bounds = {parameter: angle_prior / scale}
    reparam = Angle(
        parameters=parameter, prior_bounds=prior_bounds, scale=scale
    )
    angle = np.random.uniform(*prior_bounds[parameter], n)
    assert assert_invertibility(reparam, angle, radial=None)


@pytest.mark.parametrize(
    "angle_prior", [np.array([0, 2 * np.pi]), np.array([-np.pi, np.pi])]
)
def test_invertiblity_both_parameters(
    angle_prior, scale, assert_invertibility
):
    """Test the inverbility when using just an angle"""
    n = 100
    parameters = ["theta", "r"]
    prior_bounds = {
        parameters[0]: angle_prior / scale,
        parameters[1]: np.array([0, 5]),
    }
    reparam = Angle(
        parameters=parameters, prior_bounds=prior_bounds, scale=scale
    )
    angle = np.random.uniform(*prior_bounds[parameters[0]], n)
    radial = np.random.uniform(*prior_bounds[parameters[1]], n)
    assert assert_invertibility(reparam, angle, radial=radial)


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "value, output_x, output_y",
    [
        (-1.0, None, 0.0),
        (1.0, None, 0.0),
        (-0.5, 0.0, None),
        (0.5, 0.0, None),
    ],
)
def test_periodic_parameter(value, output_x, output_y):
    """Test a generic periodic parameter"""
    parameters = ["a"]
    prior_bounds = {"a": [-1.0, 1.0]}
    reparam = Angle(
        parameters=parameters, prior_bounds=prior_bounds, scale=None
    )
    x = parameters_to_live_point((value,), parameters)
    x_prime = parameters_to_live_point(
        (np.nan, np.nan), reparam.prime_parameters
    )
    log_j = np.zeros(x.size)
    x_out, x_prime_out, log_j_out = reparam.reparameterise(x, x_prime, log_j)

    assert_structured_arrays_equal(x_out, x)

    if output_x:
        assert x_prime_out[reparam.x] == output_x
    if output_y:
        assert x_prime_out[reparam.y] == output_y


def test_reparameterise_negative_radius(reparam):
    """Assert an error is radius if the radius is negative."""
    x = numpy_array_to_live_points(
        np.array([[1.0, -1.0]]), ["theta", "radius"]
    )
    x_prime = x.copy()
    log_j = 0.0
    reparam.radial = "radius"
    reparam.chi = None

    reparam._rescale_angle = lambda *args: (x["theta"], *args)
    reparam._rescale_radial = lambda *args: (x["radius"], *args)

    with pytest.raises(RuntimeError) as excinfo:
        Angle.reparameterise(reparam, x, x_prime, log_j)
    assert "Radius cannot be negative" in str(excinfo.value)
