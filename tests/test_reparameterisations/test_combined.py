# -*- coding: utf-8 -*-
"""
Test the combined reparameterisation class.
"""
import numpy as np

from nessai.reparameterisations import (
    CombinedReparameterisation,
    RescaleToBounds
    )
from nessai.livepoint import get_dtype


def test_init_none():
    c = CombinedReparameterisation()
    assert c.parameters == []


def test_init_w_reparam():
    c = CombinedReparameterisation(RescaleToBounds('x', [0, 1]))
    assert c.parameters == ['x']
    assert c.prime_parameters == ['x_prime']


def test_add_single_reparameterisations():
    """Test the core functionality of adding reparameterisations"""
    r = RescaleToBounds(parameters='x', prior_bounds=[0, 1])
    c = CombinedReparameterisation()
    c.add_reparameterisations(r)
    assert c.parameters == ['x']
    assert c.has_prime_prior is False


def test_add_multiple_reparameterisations(model):
    """
    Test adding multiple reparameterisations and using the reparameterisation.
    """
    r = [RescaleToBounds(parameters='x', prior_bounds=model.bounds['x'],
                         prior='uniform'),
         RescaleToBounds(parameters='y', prior_bounds=model.bounds['y'],
                         prior='uniform')]
    reparam = CombinedReparameterisation()
    reparam.add_reparameterisations(r)

    assert reparam.parameters == ['x', 'y']
    assert reparam.has_prime_prior is True

    n = 100
    x = model.new_point(N=n)
    x_prime = np.zeros([n], dtype=get_dtype(reparam.prime_parameters))
    log_j = 0

    x_re, x_prime_re, log_j_re = reparam.reparameterise(
        x, x_prime, log_j)

    assert reparam.x_prime_log_prior(x_prime_re) is not None

    np.testing.assert_array_equal(x, x_re)

    x_in = np.zeros([n], dtype=get_dtype(reparam.parameters))

    x_inv, x_prime_inv, log_j_inv = \
        reparam.inverse_reparameterise(x_in, x_prime_re, log_j)

    np.testing.assert_array_equal(x, x_inv)
    np.testing.assert_array_equal(x_prime_re, x_prime_inv)
    np.testing.assert_array_equal(log_j_re, -log_j_inv)
