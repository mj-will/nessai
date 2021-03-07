# -*- coding: utf-8 -*-
"""
Test the RescaleToBound class.
"""
import numpy as np
import pytest

from nessai.reparameterisations import RescaleToBounds
from nessai.livepoint import get_dtype


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
