# -*- coding: utf-8 -*-
"""
Test the base reparameterisation.
"""
import numpy as np
from numpy.testing import assert_equal
import pytest
from unittest.mock import create_autospec

from nessai.reparameterisations import Reparameterisation


@pytest.fixture()
def reparam():
    return create_autospec(Reparameterisation)


@pytest.mark.parametrize("name", ["x1", ["x1"]])
@pytest.mark.parametrize("prior_bounds", [[0, 1], (0, 1), {"x1": [0, 1]}])
def test_init(name, prior_bounds):
    """Test the init method with the allowed types of inputs"""
    reparam = Reparameterisation(parameters=name, prior_bounds=prior_bounds)
    assert reparam.parameters == ["x1"]
    assert reparam.prime_parameters == ["x1_prime"]
    assert_equal(reparam.prior_bounds, {"x1": np.array([0, 1])})


def test_init_infinite_bounds():
    """Test the init method with infinite prior bounds"""
    reparam = Reparameterisation(
        parameters=["x", "y"], prior_bounds={"x": [0, 1], "y": [0, np.inf]}
    )
    assert reparam.parameters == ["x", "y"]
    assert reparam.prime_parameters == ["x_prime", "y_prime"]
    assert_equal(reparam.prior_bounds["x"], [0, 1])
    assert_equal(reparam.prior_bounds["y"], [0, np.inf])


def test_infinite_bounds_error():
    """Test to ensure infinite prior bounds raise an error.

    Only applies if `requires_bounded_prior` is True.
    """

    class TestReparam(Reparameterisation):
        requires_bounded_prior = True

    with pytest.raises(RuntimeError) as excinfo:
        TestReparam(
            parameters=["x", "y"], prior_bounds={"x": [0, 1], "y": [0, np.inf]}
        )
    assert "requires finite prior" in str(excinfo.value)


def test_no_prior_bounds():
    """Test not providing prior bounds."""

    class TestReparam(Reparameterisation):
        requires_bounded_prior = False

    reparam = TestReparam(parameters=["x", "y"])
    assert reparam.prior_bounds is None


def test_no_prior_bounds_error():
    """Test missing prior bounds error.

    Only applies if `requires_bounded_prior` is True.
    """

    class TestReparam(Reparameterisation):
        requires_bounded_prior = True

    with pytest.raises(RuntimeError) as excinfo:
        TestReparam(parameters=["x", "y"])
    assert "requires prior bounds" in str(excinfo.value)


def test_parameters_error():
    with pytest.raises(TypeError) as excinfo:
        Reparameterisation(parameters={"x": [0, 1]})
    assert "Parameters must be a str or list" in str(excinfo.value)


def test_missing_bounds():
    with pytest.raises(RuntimeError) as excinfo:
        Reparameterisation(parameters=["x", "y"], prior_bounds={"x": [0, 1]})
    assert "Mismatch" in str(excinfo.value)


def test_incorrect_bounds_type():
    with pytest.raises(TypeError) as excinfo:
        Reparameterisation(parameters=["x", "y"], prior_bounds=1)
    assert "Prior bounds must be" in str(excinfo.value)


def test_incorrect_bounds_length():
    with pytest.raises(RuntimeError) as excinfo:
        Reparameterisation(parameters=["x", "y"], prior_bounds=[1, 2, 3])
    assert "Prior bounds got a list of len > 2" in str(excinfo.value)


def test_methods_not_implemented():
    """Test to ensure class fails if user does not define the methods"""
    reparam = Reparameterisation(parameters="x", prior_bounds=[0, 1])

    with pytest.raises(NotImplementedError):
        reparam.reparameterise(None, None, None)

    with pytest.raises(NotImplementedError):
        reparam.inverse_reparameterise(None, None, None)


def test_update(reparam):
    """Assert the default update method can be called and does not raised an
    error.
    """
    x = np.array((1, 2), dtype=[("x", "f8"), ("y", "f8")])
    Reparameterisation.update(reparam, x)
