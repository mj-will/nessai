from unittest.mock import create_autospec

import numpy as np
import pytest

from nessai.livepoint import dict_to_live_points, empty_structured_array
from nessai.reparameterisations.discrete import Dequantise
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture
def reparam(rng):
    return create_autospec(Dequantise, rng=rng)


def test_dequantise_init():
    reparam = Dequantise(parameters=["a"], prior_bounds={"a": [0, 10]})
    assert reparam.has_pre_rescaling is True
    assert reparam.has_prime_prior is False

    # Bounds should be one more than the max
    assert reparam.bounds["a"][0] == 0
    assert reparam.bounds["a"][1] == 11


def test_pre_rescaling(reparam):
    x = np.array([0, 0, 0, 1, 1, 1])
    out, log_j = Dequantise.pre_rescaling(reparam, x)
    assert all((out[:3] > 0) & (out[:3] < 1))
    assert all((out[3:] > 1) & (out[3:] < 2))
    np.testing.assert_equal(log_j, 0.0)


def test_pre_rescaling_inv(reparam):
    x = np.array([0.5, 1.1, 2.3])
    out, log_j = Dequantise.pre_rescaling_inv(reparam, x)
    np.testing.assert_array_equal(out, np.array([0, 1, 2]))
    np.testing.assert_equal(log_j, 0.0)


def test_dequantise_integration():
    values = [0, 1, 2, 4, 8]
    x = dict_to_live_points({"a": np.random.choice(values, size=10)})

    reparam = Dequantise(
        parameters="a",
        prior_bounds={"a": [0, 8]},
        rescale_bounds=[0, 1],
    )

    x_prime = empty_structured_array(len(x), names=reparam.prime_parameters)
    x_out, x_prime, log_j_out = reparam.reparameterise(
        x, x_prime, np.zeros(len(x))
    )
    assert_structured_arrays_equal(x_out, x)

    assert not np.array_equal(x_prime["a_prime"], x["a"])
    assert all((x_prime["a_prime"] > 0) & (x_prime["a_prime"] < 1))
    # Should have 10 unique values
    assert len(np.unique(x_prime["a_prime"]) == len(x))

    x_re, x_prime_re, log_j_re = reparam.inverse_reparameterise(
        x_out.copy(), x_prime.copy(), np.zeros(len(x))
    )
    assert_structured_arrays_equal(x_prime_re, x_prime)

    assert_structured_arrays_equal(x_re, x)
    np.testing.assert_array_equal(-log_j_re, log_j_out)
