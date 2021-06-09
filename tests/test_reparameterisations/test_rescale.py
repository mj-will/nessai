# -*- coding: utf-8 -*-
"""
Test the Rescale class.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.livepoint import numpy_array_to_live_points
from nessai.reparameterisations import Rescale


@pytest.fixture()
def reparam():
    return create_autospec(Rescale)


@pytest.mark.parametrize('scale', [2, 2.0, [1, 2], {'x': 1, 'y': 2}])
def test_init(scale):
    """Test the init method with different input types"""
    parameters = ['x', 'y']
    prior_bounds = {'x': [-1, 1], 'y': [-1, 1]}

    reparam = \
        Rescale(parameters=parameters, scale=scale, prior_bounds=prior_bounds)

    assert not set(reparam.scale.keys()) - set(parameters)
    assert isinstance(reparam.scale['x'], float)


@pytest.mark.parametrize('n', [1, 2])
def test_reparameterise(reparam, n):
    """Test the reparameterise method"""
    reparam.parameters = ['x', 'y']
    reparam.prime_parameters = ['x_prime', 'y_prime']
    reparam.scale = {'x': -2.0, 'y': 4.0}
    x = numpy_array_to_live_points(np.ones((n, 2)), reparam.parameters)
    x_prime = numpy_array_to_live_points(
        np.zeros((n, 2)), reparam.prime_parameters)
    log_j = np.zeros(n)

    x_out, x_prime_out, log_j_out = \
        Rescale.reparameterise(reparam, x, x_prime, log_j)

    assert np.array_equal(x, x_out)
    assert np.array_equal(log_j_out, -np.log(8 * np.ones(n)))
    assert (x_prime_out['x_prime'] == -0.5).all()
    assert (x_prime_out['y_prime'] == 0.25).all()


@pytest.mark.parametrize('scale', [1e60, 1e-60])
def test_reparameterise_overflow(reparam, scale):
    """Test the reparameterise method with very small and large scales.

    Checks precision to 14 decimal places.
    """
    reparam.parameters = ['x']
    reparam.prime_parameters = ['x_prime']
    reparam.scale = {'x': scale}
    x_array = np.arange(100.0, dtype=float)
    x = numpy_array_to_live_points(scale * x_array[:, np.newaxis],
                                   reparam.parameters)
    x_prime = numpy_array_to_live_points(np.ones((x_array.size, 1)),
                                         reparam.prime_parameters)
    log_j = np.zeros(x.size)

    x_out, x_prime_out, log_j_out = \
        Rescale.reparameterise(reparam, x, x_prime, log_j)

    np.testing.assert_array_almost_equal(x_array, x_prime_out['x_prime'],
                                         decimal=14)
    assert (log_j == -np.log(scale)).all()


@pytest.mark.parametrize('n', [1, 2])
def test_inverse_reparameterise(reparam, n):
    """Test the inverse reparameterise method"""
    reparam.parameters = ['x', 'y']
    reparam.prime_parameters = ['x_prime', 'y_prime']
    reparam.scale = {'x': -2.0, 'y': 4.0}
    x = numpy_array_to_live_points(np.zeros((n, 2)), reparam.parameters)
    x_prime = numpy_array_to_live_points(
        np.ones((n, 2)), reparam.prime_parameters)
    x_prime['x_prime'] *= -1
    log_j = np.zeros(n)

    x_out, x_prime_out, log_j_out = \
        Rescale.inverse_reparameterise(reparam, x, x_prime, log_j)

    assert np.array_equal(x_prime, x_prime_out)
    assert np.array_equal(log_j_out, np.log(8 * np.ones(n)))
    assert (x_out['x'] == 2.0).all()
    assert (x_out['y'] == 4.0).all()


@pytest.mark.parametrize('scale', [1e60, 1e-60])
def test_inverse_reparameterise_overflow(reparam, scale):
    """Test the inverse_reparameterise method with very small and large scales.
    """
    reparam.parameters = ['x']
    reparam.prime_parameters = ['x_prime']
    reparam.scale = {'x': scale}
    x_array = np.arange(100.0, dtype=float)
    x = numpy_array_to_live_points(np.ones((x_array.size, 1)),
                                   reparam.parameters)
    x_prime = numpy_array_to_live_points(x_array[:, np.newaxis],
                                         reparam.prime_parameters)
    log_j = np.zeros(x.size)

    x_out, x_prime_out, log_j_out = \
        Rescale.inverse_reparameterise(reparam, x, x_prime, log_j)

    np.testing.assert_array_equal(x_array * scale, x_out['x'])
    assert (log_j == np.log(scale)).all()


def test_init_no_scale():
    """Make sure an error is raised if the scale is not given"""
    with pytest.raises(RuntimeError) as excinfo:
        Rescale(scale=None)
    assert 'Must specify a scale!' in str(excinfo.value)


@pytest.mark.parametrize('scale', [[1], [1, 2, 3]])
def test_init_incorrect_scale_list(scale):
    """Make sure an error is raised if the scale is the incorrect length"""
    parameters = ['x', 'y']
    prior_bounds = {'x': [-1, 1], 'y': [-1, 1]}

    with pytest.raises(RuntimeError) as excinfo:
        Rescale(parameters=parameters, scale=scale, prior_bounds=prior_bounds)

    assert 'different length' in str(excinfo.value)


@pytest.mark.parametrize('scale', [{'x': 1}, {'x': 1, 'y': 1, 'z': 1}])
def test_init_incorrect_scale_dict(scale):
    """Make sure an error is raised if the scale keys to not match the \
            parameters.
    """
    parameters = ['x', 'y']
    prior_bounds = {'x': [-1, 1], 'y': [-1, 1]}

    with pytest.raises(RuntimeError) as excinfo:
        Rescale(parameters=parameters, scale=scale, prior_bounds=prior_bounds)

    assert 'Mismatched parameters' in str(excinfo.value)


def test_init_incorrect_scale_type():
    """Make sure an error is raised if the scale is the incorrect type"""
    parameters = ['x', 'y']
    prior_bounds = {'x': [-1, 1], 'y': [-1, 1]}

    with pytest.raises(TypeError) as excinfo:
        Rescale(parameters=parameters, scale='1', prior_bounds=prior_bounds)

    assert 'Scale input must be' in str(excinfo.value)
