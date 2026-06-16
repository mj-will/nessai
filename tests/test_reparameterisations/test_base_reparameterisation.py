# -*- coding: utf-8 -*-
"""
Test the base reparameterisation.
"""

import copy
from unittest.mock import create_autospec

import numpy as np
import pytest
from numpy.testing import assert_equal

from nessai.livepoint import empty_structured_array
from nessai.reparameterisations import Reparameterisation


@pytest.fixture()
def reparam():
    return create_autospec(Reparameterisation)


@pytest.mark.parametrize("name", ["x1", ["x1"]])
@pytest.mark.parametrize("prior_bounds", [[0, 1], (0, 1), {"x1": [0, 1]}])
def test_init(name, prior_bounds):
    """Test the init method with the allowed types of inputs"""
    reparam = Reparameterisation(parameters=name, prior_bounds=prior_bounds)
    assert reparam.input_parameters == ["x1"]
    assert reparam.output_parameters == ["x1_prime"]
    assert_equal(reparam.prior_bounds, {"x1": np.array([0, 1])})


def test_init_infinite_bounds():
    """Test the init method with infinite prior bounds"""
    reparam = Reparameterisation(
        parameters=["x", "y"], prior_bounds={"x": [0, 1], "y": [0, np.inf]}
    )
    assert reparam.input_parameters == ["x", "y"]
    assert reparam.output_parameters == ["x_prime", "y_prime"]
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
    class TestReparam(Reparameterisation):
        requires_bounded_prior = True

    with pytest.raises(RuntimeError) as excinfo:
        TestReparam(parameters=["x", "y"], prior_bounds={"x": [0, 1]})
    assert "Mismatch" in str(excinfo.value)


def test_missing_bounds_allowed_for_auxiliary_parameters():
    reparam = Reparameterisation(
        parameters=["x", "aux"], prior_bounds={"x": [0, 1]}
    )
    assert_equal(reparam.prior_bounds, {"x": np.array([0, 1])})


def test_conflicting_parameters_and_input_parameters():
    with pytest.raises(
        RuntimeError, match="Received conflicting values for `parameters`"
    ):
        Reparameterisation(
            parameters=["x"],
            input_parameters=["y"],
            prior_bounds={"y": [0, 1]},
        )


def test_persistent_parameters_must_be_subset():
    with pytest.raises(
        RuntimeError,
        match="Persistent parameters must be a subset of the input",
    ):
        Reparameterisation(
            parameters=["x"],
            persistent_parameters=["y"],
            prior_bounds={"x": [0, 1]},
        )


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


def test_output_parameters():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    reparam.auxiliary_parameters = ["x_aux"]
    assert reparam.output_parameters == ["x_prime"]
    assert reparam.x_output_parameters == ["x", "x_aux"]


def test_format_parameters_invalid_type():
    with pytest.raises(TypeError, match="Parameters must be a string"):
        Reparameterisation._format_parameters(1)


def test_update(reparam):
    """Assert the default update method can be called and does not raised an
    error.
    """
    x = np.array((1, 2), dtype=[("x", "f8"), ("y", "f8")])
    Reparameterisation.update(reparam, x)


def test_reset(reparam):
    expected = copy.deepcopy(reparam)
    Reparameterisation.reset(reparam)
    assert expected == reparam


def test_resolve_forward_input_spaces():
    reparam = Reparameterisation(
        input_parameters=["x", "x_prime", "missing"],
        persistent_parameters=["x", "x_prime"],
        prior_bounds={"x": [0, 1]},
    )

    missing = reparam.resolve_forward_input_spaces(
        available_parameters=["x", "y"],
        available_prime_parameters=["x_prime", "y_prime"],
    )

    assert missing == ["missing"]
    assert reparam.x_input_parameters == ["x"]
    assert reparam.x_prime_input_parameters == ["x_prime"]
    assert reparam.x_persistent_parameters == ["x"]
    assert reparam.x_prime_persistent_parameters == ["x_prime"]


def test_resolve_inverse_input_spaces():
    reparam = Reparameterisation(
        parameters=["x"],
        inverse_input_parameters=["y", "y_prime", "missing"],
        prior_bounds={"x": [0, 1]},
    )

    missing = reparam.resolve_inverse_input_spaces(
        available_parameters=["x", "y"],
        available_prime_parameters=["x_prime", "y_prime"],
    )

    assert missing == ["missing"]
    assert reparam.x_inverse_input_parameters == ["y"]
    assert reparam.x_prime_inverse_input_parameters == ["y_prime"]


def test_get_parameter_value_from_x():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    x = empty_structured_array(2, names=["x"])
    x["x"] = np.array([1.0, 2.0])

    out = reparam.get_parameter_value("x", x)

    np.testing.assert_array_equal(out, x["x"])


def test_get_parameter_value_from_x_prime():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    reparam._x_prime_input_parameters = ["x"]
    x = empty_structured_array(2, names=["x"])
    x["x"] = np.array([1.0, 2.0])
    x_prime = empty_structured_array(2, names=["x"])
    x_prime["x"] = np.array([3.0, 4.0])

    out = reparam.get_parameter_value("x", x, x_prime=x_prime)

    np.testing.assert_array_equal(out, x_prime["x"])


def test_get_parameter_value_from_x_prime_missing_array():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    reparam._x_prime_input_parameters = ["x"]
    x = empty_structured_array(2, names=["x"])

    with pytest.raises(RuntimeError, match="no x_prime array was provided"):
        reparam.get_parameter_value("x", x)


def test_set_parameter_value_in_x():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    x = empty_structured_array(2, names=["x"])

    x_out, x_prime_out = reparam.set_parameter_value(
        "x", np.array([1.0, 2.0]), x
    )

    np.testing.assert_array_equal(x_out["x"], np.array([1.0, 2.0]))
    assert x_prime_out is None


def test_set_parameter_value_in_x_prime():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    reparam._x_prime_input_parameters = ["x"]
    x = empty_structured_array(2, names=["x"])
    x_prime = empty_structured_array(2, names=["x"])

    x_out, x_prime_out = reparam.set_parameter_value(
        "x", np.array([3.0, 4.0]), x, x_prime=x_prime
    )

    np.testing.assert_array_equal(x_prime_out["x"], np.array([3.0, 4.0]))
    assert x_out is x


def test_set_parameter_value_in_x_prime_missing_array():
    reparam = Reparameterisation(parameters=["x"], prior_bounds={"x": [0, 1]})
    reparam._x_prime_input_parameters = ["x"]
    x = empty_structured_array(2, names=["x"])

    with pytest.raises(RuntimeError, match="no x_prime array was provided"):
        reparam.set_parameter_value("x", np.array([1.0, 2.0]), x)
