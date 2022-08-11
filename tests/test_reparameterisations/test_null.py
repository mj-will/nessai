# -*- coding: utf-8 -*-
"""
Test the null reparameterisation
"""
import numpy as np

from nessai.reparameterisations import NullReparameterisation
from nessai.livepoint import empty_structured_array, numpy_array_to_live_points
from nessai.utils.testing import (
    assert_structured_arrays_equal as assert_array_equal,
)


def test_invertiblity():
    """Ensure the reparameterisation does not change the values"""
    reparam = NullReparameterisation(parameters=["x", "y"])
    n = 100
    x = numpy_array_to_live_points(np.random.rand(n, 2), reparam.parameters)
    x_prime = empty_structured_array(n, names=reparam.prime_parameters)
    log_j = 0

    x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

    assert_array_equal(x, x_re)
    assert_array_equal(x, x_prime_re)

    x_in = empty_structured_array(n, names=reparam.parameters)
    x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
        x_in, x_prime_re, log_j
    )

    assert_array_equal(x, x_inv)
    assert_array_equal(x, x_prime_inv)
    assert log_j == log_j_re == log_j_inv
