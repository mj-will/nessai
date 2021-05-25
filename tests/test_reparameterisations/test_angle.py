# -*- coding: utf-8 -*-
"""
Test the Angle reparameterisation
"""
import numpy as np
import pytest

from nessai.reparameterisations import Angle
from nessai.livepoint import get_dtype

scales = [1.0, 2.0]


@pytest.fixture(params=scales, scope='function')
def scale(request):
    return request.param


@pytest.fixture(scope='function')
def assert_invertibility(model, n=100):
    def test_invertibility(reparam, angle, radial=None):

        x = np.zeros([n], dtype=get_dtype(reparam.parameters))
        x_prime = np.zeros([n], dtype=get_dtype(reparam.prime_parameters))
        log_j = 0

        x[reparam.angle] = angle
        if radial is not None:
            x[reparam.radial] = radial

        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(
            x, x_prime, log_j)

        np.testing.assert_array_equal(x[reparam.angle], x_re[reparam.angle])
        if radial is not None:
            np.testing.assert_array_equal(x[reparam.radial],
                                          x_re[reparam.radial])

        x_in = np.zeros([n], dtype=get_dtype(reparam.parameters))

        x_inv, x_prime_inv, log_j_inv = \
            reparam.inverse_reparameterise(x_in, x_prime_re, log_j)

        np.testing.assert_array_almost_equal(x[reparam.angle],
                                             x_inv[reparam.angle])
        if radial is not None:
            np.testing.assert_array_almost_equal(x[reparam.radial],
                                                 x_inv[reparam.radial])

        np.testing.assert_array_equal(x_prime_re, x_prime_inv)
        np.testing.assert_array_almost_equal(log_j_re, -log_j_inv)

        return True

    return test_invertibility


def test_angle_parameter():
    """Test init with just an angle parameter"""
    parameter = 'theta'
    prior_bounds = {parameter: np.array([0, 2 * np.pi])}
    reparam = Angle(parameters=parameter, prior_bounds=prior_bounds)

    assert reparam.chi is not False
    assert hasattr(reparam.chi, 'rvs')
    assert reparam._zero_bound is True
    assert reparam.has_prime_prior is False

    assert reparam.angle == parameter
    assert reparam.radial == (parameter + '_radial')
    assert reparam.radius == (parameter + '_radial')
    assert reparam.x == (parameter + '_x')
    assert reparam.y == (parameter + '_y')


def test_both_parameters():
    """Test init with just an angle and radial parameter"""
    parameters = ['theta', 'r']
    prior_bounds = {parameters[0]: np.array([0, 2 * np.pi]),
                    parameters[1]: np.array([0, 5])}

    reparam = Angle(parameters=parameters, prior_bounds=prior_bounds)

    assert reparam.chi is False
    assert reparam._zero_bound is True
    assert reparam.has_prime_prior is False

    assert reparam.angle == parameters[0]
    assert reparam.radial == parameters[1]


@pytest.mark.parametrize('angle_prior', [np.array([0, 2 * np.pi]),
                                         np.array([-np.pi, np.pi])])
def test_invertiblity_single_parameter(angle_prior, scale,
                                       assert_invertibility):
    """Test the inverbility when using just an angle"""
    n = 100
    parameter = 'theta'
    prior_bounds = {parameter: angle_prior / scale}
    reparam = Angle(parameters=parameter, prior_bounds=prior_bounds,
                    scale=scale)
    angle = np.random.uniform(*prior_bounds[parameter], n)
    assert assert_invertibility(reparam, angle, radial=None)


@pytest.mark.parametrize('angle_prior', [np.array([0, 2 * np.pi]),
                                         np.array([-np.pi, np.pi])])
def test_invertiblity_both_parameters(angle_prior,
                                      scale, assert_invertibility):
    """Test the inverbility when using just an angle"""
    n = 100
    parameters = ['theta', 'r']
    prior_bounds = {parameters[0]: angle_prior / scale,
                    parameters[1]: np.array([0, 5])}
    reparam = Angle(parameters=parameters, prior_bounds=prior_bounds,
                    scale=scale)
    angle = np.random.uniform(*prior_bounds[parameters[0]], n)
    radial = np.random.uniform(*prior_bounds[parameters[1]], n)
    assert assert_invertibility(reparam, angle, radial=radial)
