# -*- coding: utf-8 -*-
"""
Test the RescaleToBound class.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.reparameterisations import RescaleToBounds
from nessai.livepoint import get_dtype, numpy_array_to_live_points


@pytest.fixture
def reparam():
    return create_autospec(RescaleToBounds)


@pytest.fixture()
def reparameterisation(model):
    def _get_reparameterisation(kwargs):
        return RescaleToBounds(parameters=model.names,
                               prior_bounds=model.bounds,
                               **kwargs)
    return _get_reparameterisation


@pytest.fixture(scope='function')
def assert_invertibility(model, n=100):
    def test_invertibility(reparam):
        x = model.new_point(N=n)
        x_prime = np.zeros([n], dtype=get_dtype(reparam.prime_parameters))
        log_j = 0

        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(
            x, x_prime, log_j)

        np.testing.assert_array_equal(x, x_re)

        x_in = np.zeros([n], dtype=get_dtype(reparam.parameters))

        x_inv, x_prime_inv, log_j_inv = \
            reparam.inverse_reparameterise(x_in, x_prime_re, log_j)

        np.testing.assert_array_equal(x, x_inv)
        np.testing.assert_array_equal(x_prime_re, x_prime_inv)
        np.testing.assert_array_equal(log_j_re, -log_j_inv)

        return True

    return test_invertibility


@pytest.mark.parametrize('rescale_bounds', [None, [0, 1]])
def test_rescale_bounds(reparameterisation, assert_invertibility,
                        rescale_bounds):
    """Test the different options for rescale to bounds"""
    reparam = reparameterisation({'rescale_bounds': rescale_bounds})
    if rescale_bounds is None:
        rescale_bounds = {p: [-1, 1] for p in reparam.parameters}
    elif isinstance(rescale_bounds, list):
        rescale_bounds = {p: rescale_bounds for p in reparam.parameters}

    assert reparam.rescale_bounds == rescale_bounds
    assert assert_invertibility(reparam)


@pytest.mark.parametrize('boundary_inversion', [False,
                                                True,
                                                ['x']])
def test_boundary_inversion(reparameterisation, assert_invertibility,
                            boundary_inversion):
    """Test the different options for rescale to bounds"""
    reparam = reparameterisation({'boundary_inversion': boundary_inversion})

    assert assert_invertibility(reparam)


def test_set_bounds(reparam):
    """Test the set bounds method."""
    reparam.parameters = ['x']
    reparam.rescale_bounds = {'x': np.array([-1, 1])}
    reparam.pre_rescaling = lambda x: (x / 2, np.zeros_like(x))
    reparam.offsets = {'x': 1}
    RescaleToBounds.set_bounds(reparam, {'x': np.array([-10, 10])})
    np.testing.assert_array_equal(reparam.pre_prior_bounds['x'], [-5, 5])
    np.testing.assert_array_equal(reparam.bounds['x'], [-6, 4])


@pytest.mark.integration_test
def test_update_prime_prior_bounds_integration():
    """Assert the prime prior bounds are correctly computed"""
    rescaling = (
        lambda x: (x / 2, np.zeros_like(x)),
        lambda x: (2 * x, np.zeros_like(x)),
    )
    reparam = RescaleToBounds(
        parameters=['x'], prior_bounds=[1000, 1001], prior='uniform',
        pre_rescaling=rescaling, offset=True,
    )
    np.testing.assert_equal(reparam.offsets['x'], 500.25)
    np.testing.assert_array_equal(reparam.prior_bounds['x'], [1000, 1001])
    np.testing.assert_array_equal(reparam.pre_prior_bounds['x'], [500, 500.5])
    np.testing.assert_array_equal(reparam.bounds['x'], [-0.25, 0.25])
    np.testing.assert_array_equal(
        reparam.prime_prior_bounds['x_prime'], [-1, 1]
    )

    x_prime = numpy_array_to_live_points(
        np.array([[-2], [-1], [0.5], [1], [10]]), ['x_prime']
    )
    log_prior = reparam.x_prime_log_prior(x_prime)
    expected = np.array([-np.inf, 0, 0, 0, -np.inf])
    np.testing.assert_equal(log_prior, expected)
