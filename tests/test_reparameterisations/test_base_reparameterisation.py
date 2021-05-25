# -*- coding: utf-8 -*-
"""
Test the base reparameterisation.
"""
import numpy as np
from numpy.testing import assert_equal
import pytest

from nessai.reparameterisations import Reparameterisation


@pytest.mark.parametrize('name', ['x1', ['x1']])
@pytest.mark.parametrize('prior_bounds', [[0, 1], (0, 1), {'x1': [0, 1]}])
def test_init(name, prior_bounds):
    """Test the init method with the allowed types of inputs"""
    reparam = Reparameterisation(parameters=name, prior_bounds=prior_bounds)
    assert reparam.parameters == ['x1']
    assert reparam.prime_parameters == ['x1_prime']
    assert_equal(reparam.prior_bounds, {'x1': np.array([0, 1])})


def test_parameters_error():
    with pytest.raises(TypeError) as excinfo:
        Reparameterisation(parameters={'x': [0, 1]})
    assert 'Parameters must be a str or list' in str(excinfo.value)


def test_missing_bounds():
    with pytest.raises(RuntimeError) as excinfo:
        Reparameterisation(parameters=['x', 'y'], prior_bounds={'x': [0, 1]})
    assert 'Mismatch' in str(excinfo.value)


def test_incorrect_bounds_type():
    with pytest.raises(TypeError) as excinfo:
        Reparameterisation(parameters=['x', 'y'], prior_bounds=1)
    assert 'Prior bounds must be' in str(excinfo.value)


def test_methods_not_implemented():
    """Test to ensure class fails if user does not define the methods"""
    reparam = Reparameterisation(parameters='x', prior_bounds=[0, 1])

    with pytest.raises(NotImplementedError):
        reparam.reparameterise(None, None, None)

    with pytest.raises(NotImplementedError):
        reparam.inverse_reparameterise(None, None, None)
