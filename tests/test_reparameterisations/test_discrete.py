"""Test reparameterisations for discrete variables"""
from unittest.mock import create_autospec

from nessai.livepoint import numpy_array_to_live_points, empty_structured_array
from nessai.reparameterisations.discrete import Dequantise
from nessai.utils.testing import assert_structured_arrays_equal
import numpy as np
import pytest


@pytest.fixture
def dequant():
    r = create_autospec(Dequantise)
    r.include_logit = False
    return r


def test_reparameterise(dequant):
    """Assert values are correctly dequantised"""
    x = numpy_array_to_live_points(np.array([[1, 2, 3]]).T, ["x"])
    x_prime = empty_structured_array(x.size, ["x_dequant"])
    log_j = np.zeros(x.size)

    dequant.parameters = ["x"]
    dequant.prime_parameters = ["x_dequant"]
    dequant.prior_bounds = {"x": [1, 3]}

    Dequantise.reparameterise(dequant, x, x_prime, log_j)

    assert np.all(x_prime["x_dequant"] < 1.0) and np.all(
        x_prime["x_dequant"] > 0.0
    )
    assert np.all(x_prime["x_dequant"] > ((x["x"] - 1.0) / 3.0))


def test_inverse_reparameterise(dequant):
    """Assert values are correctly quantised"""
    x_prime = numpy_array_to_live_points(
        np.array([[0.2, 0.5, 0.9]]).T, ["x_dequant"]
    )
    x = empty_structured_array(x_prime.size, ["x"])
    log_j = np.zeros(x.size)
    expected = np.array([1, 2, 3])

    dequant.parameters = ["x"]
    dequant.prime_parameters = ["x_dequant"]
    dequant.prior_bounds = {"x": [1, 3]}

    Dequantise.inverse_reparameterise(dequant, x, x_prime, log_j)
    np.testing.assert_array_equal(x["x"], expected)


@pytest.mark.integration_test
@pytest.mark.parametrize("include_logit", [True, False])
def test_invertible(include_logit):
    """Assert the reparameterisation is invertible"""
    n = 5
    names = ["x", "y"]
    prior_bounds = {"x": [2, 8], "y": [0, 10]}
    reparam = Dequantise(
        parameters=names,
        prior_bounds=prior_bounds,
        include_logit=include_logit,
    )

    x = numpy_array_to_live_points(
        np.random.randint(low=[2, 0], high=[8, 10], size=(n, len(names))),
        names=reparam.parameters,
    )
    x_prime = empty_structured_array(n, reparam.prime_parameters)
    log_j = np.zeros(n)

    x_forward, x_prime_forward, log_j_forward = reparam.reparameterise(
        x, x_prime, log_j
    )
    x_re, x_prime_re, log_j_re = reparam.inverse_reparameterise(
        x_forward, x_prime_forward, log_j_forward
    )

    # Assert input does not change
    assert_structured_arrays_equal(x_forward, x)
    # Assert is invertible
    assert_structured_arrays_equal(x_re, x)
