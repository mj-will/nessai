# -*- coding: utf-8 -*-
"""
Tests for `nessai.model`
"""

import datetime
import logging
from unittest.mock import MagicMock, call, create_autospec, patch

import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from scipy.stats import norm

from nessai import config
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model, OneDimensionalModelError
from nessai.utils.errors import RNGNotSetError, RNGSetError
from nessai.utils.multiprocessing import (
    initialise_pool_variables,
    log_likelihood_wrapper,
)
from nessai.utils.testing import (
    assert_structured_arrays_equal,
)


class EmptyModel(Model):
    def log_prior(self, x):
        return None

    def log_likelihood(self, x):
        return None


class BasicModel(Model):
    def __init__(self):
        self.bounds = {"x": [-5, 5], "y": [-5, 5]}
        self.names = ["x", "y"]

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        log_l = np.ones(x.size)
        for pn in self.names:
            log_l += norm.logpdf(x[pn])
        return log_l


@pytest.fixture
def model():
    return create_autospec(Model, _pool_configured=False)


@pytest.fixture
def live_point(integration_model, rng):
    if integration_model.rng is None:
        integration_model.set_rng(rng)
    return integration_model.new_point()


@pytest.fixture
def live_points(integration_model, rng):
    if integration_model.rng is None:
        integration_model.set_rng(rng)
    return integration_model.new_point(10)


def test_names(model):
    """Assert names returns the correct value."""
    model._names = ["x", "y"]
    assert Model.names.__get__(model) is model._names


def test_names_setter(model):
    """Assert names is correctly set"""
    Model.names.__set__(model, ["x", "y"])
    assert model._names == ["x", "y"]


def test_names_invalid_type(model):
    """Assert an error is raised if `names` is not a list."""
    with pytest.raises(TypeError) as excinfo:
        Model.names.__set__(model, True)
    assert "`names` must be a list" in str(excinfo.value)


def test_names_empty_list(model):
    """Assert an error is raised if `names` is empty."""
    with pytest.raises(ValueError) as excinfo:
        Model.names.__set__(model, [])
    assert "`names` list is empty" in str(excinfo.value)


def test_names_1d_list(model):
    """Assert an error is raised if `names` is 1d"""
    with pytest.raises(OneDimensionalModelError) as excinfo:
        Model.names.__set__(model, ["x"])
    assert "names list has length 1" in str(excinfo.value)


def test_bounds(model):
    """Assert bounds returns the correct value."""
    model._bounds = {"x": [-1, 1], "y": [-1, 1]}
    assert Model.bounds.__get__(model) is model._bounds


def test_bounds_setter(model):
    """Assert bounds is correctly set"""
    Model.bounds.__set__(model, {"x": [-1, 1], "y": [-2, 2]})
    assert list(model._bounds.keys()) == ["x", "y"]
    np.testing.assert_array_equal(model._bounds["x"], [-1, 1])
    np.testing.assert_array_equal(model._bounds["y"], [-2, 2])


def test_bounds_invalid_type(model):
    """Assert an error is raised if `bounds` is not a dictionary."""
    with pytest.raises(TypeError) as excinfo:
        Model.bounds.__set__(model, True)
    assert "`bounds` must be a dictionary" in str(excinfo.value)


def test_bounds_1d(model):
    """Assert an error is raised if `bounds` is 1d"""
    with pytest.raises(OneDimensionalModelError) as excinfo:
        Model.bounds.__set__(model, {"x": [0, 1]})
    assert "bounds dictionary has length 1" in str(excinfo.value)


@pytest.mark.parametrize("b", [[1], [1, 2, 3]])
def test_bounds_incorrect_length(model, b):
    """Assert an error is raised if `bounds` is 1d"""
    with pytest.raises(ValueError) as excinfo:
        Model.bounds.__set__(model, {"x": b, "y": [0, 1]})
    assert "Each entry in `bounds` must have length 2" in str(excinfo.value)


def test_dims(model):
    """Ensure dims are correct"""
    model.names = ["x", "y"]
    assert Model.dims.__get__(model) == 2


def test_dims_no_names(model):
    """Test the behaviour dims when names is empty"""
    model.names = []
    assert Model.dims.__get__(model) is None


def test_set_upper_lower(model):
    """Assert the upper and lower bounds are set correctly."""
    model.names = ["y", "x"]
    model.bounds = {"x": [0, 1], "y": [-1, 2]}
    Model._set_upper_lower(model)
    np.testing.assert_array_equal(model._lower, np.array([-1, 0]))
    np.testing.assert_array_equal(model._upper, np.array([2, 1]))


def test_lower_bounds(model):
    """Check the lower bounds are correctly set"""

    def func():
        model._lower = np.array([-1, -1])

    model._set_upper_lower = MagicMock(side_effect=func)
    model._lower = None
    bounds = Model.lower_bounds.__get__(model)
    model._set_upper_lower.assert_called_once()
    np.testing.assert_array_equal(bounds, np.array([-1, -1]))


def test_upper_bounds(model):
    """Check the upper bounds are correctly set"""

    def func():
        model._upper = np.array([1.0, 1.0])

    model._set_upper_lower = MagicMock(side_effect=func)
    model._upper = None
    bounds = Model.upper_bounds.__get__(model)
    model._set_upper_lower.assert_called_once()
    np.testing.assert_array_equal(bounds, np.array([1, 1]))


def test_discrete_parameters(model):
    value = ["a", "b"]
    model._discrete_parameters = value
    assert Model.discrete_parameters.__get__(model) == value


def test_discrete_parameters_setter(model, caplog):
    value = ["a", "b"]
    Model.discrete_parameters.__set__(model, value)
    assert model._discrete_parameters == value
    assert "discrete parameters is experimental" in str(caplog.text)


@pytest.mark.parametrize(
    "value, expected", [(["a", "b"], True), (None, False)]
)
def test_has_discrete_parameters(model, value, expected):
    model._discrete_parameters = value
    assert Model.has_discrete_parameters.__get__(model) is expected


@pytest.mark.parametrize("value", [True, False])
def test_vectorised_likelihood(model, value):
    """Assert the correct value is stored if allow_vectorised is True"""
    model._vectorised_likelihood = None
    model.allow_vectorised = True
    model.new_point = MagicMock(
        side_effect=[
            np.random.rand(1).astype([("x", "f8")]) for _ in range(10)
        ]
    )

    with patch("nessai.model.check_vectorised_function", return_value=value):
        out = Model.vectorised_likelihood.__get__(model)

    assert model._vectorised_likelihood is value
    assert out is value


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_vectorised_likelihood_not_vectorised_error(model, error):
    """
    Assert the value is False if the likelihood is not vectorised and raises
    an error.
    """

    def dummy_likelihood(x):
        if hasattr(x, "__len__"):
            raise error
        else:
            return np.log(np.random.rand())

    model._vectorised_likelihood = None
    model.log_likelihood = MagicMock(side_effect=dummy_likelihood)
    model.new_point = MagicMock(return_value=np.random.rand(10))

    out = Model.vectorised_likelihood.__get__(model)
    assert model._vectorised_likelihood is False
    assert out is False


def test_vectorised_likelihood_allow_vectorised_false(model):
    """Assert vectorised_likelihood is False if allow_vectorised is False"""
    model.allow_vectorised = False
    model.log_likelihood = MagicMock()
    model._vectorised_likelihood = None
    out = Model.vectorised_likelihood.__get__(model)
    model.log_likelihood.assert_not_called()
    assert out is False


def test_vectorised_likelihood_setter(model):
    """Assert the setter sets the correct variable."""
    Model.vectorised_likelihood.__set__(model, "test")
    assert model._vectorised_likelihood == "test"


@pytest.mark.parametrize("check_value", [True, False])
@pytest.mark.parametrize("allow_vectorised", [True, False])
def test_vectorised_log_prior(model, check_value, allow_vectorised):
    model._vectorised_prior = None
    model.allow_vectorised_prior = allow_vectorised

    with patch(
        "nessai.model.check_vectorised_function", return_value=check_value
    ):
        out = Model.vectorised_prior.__get__(model)

    assert out is (check_value and allow_vectorised)
    assert model._vectorised_prior is (check_value and allow_vectorised)


def test_vectorised_prior_setter(model):
    """Assert the setter sets the correct value"""
    Model.vectorised_prior.__set__(model, "test")
    assert model._vectorised_prior == "test"


@pytest.mark.parametrize("check_value", [True, False])
@pytest.mark.parametrize("allow_vectorised", [True, False])
def test_vectorised_log_prior_unit_hypercube(
    model, check_value, allow_vectorised
):
    model._vectorised_prior_unit_hypercube = None
    model.allow_vectorised_prior = allow_vectorised

    with patch(
        "nessai.model.check_vectorised_function", return_value=check_value
    ):
        out = Model.vectorised_prior_unit_hypercube.__get__(model)

    assert out is (check_value and allow_vectorised)
    assert model._vectorised_prior_unit_hypercube is (
        check_value and allow_vectorised
    )


def test_vectorised_prior_unit_setter(model):
    """Assert the setter sets the correct value"""
    Model.vectorised_prior_unit_hypercube.__set__(model, "test")
    assert model._vectorised_prior_unit_hypercube == "test"


def test_in_bounds(model):
    """Test the `in_bounds` method.

    Tests both finite and infinite prior bounds.
    """
    x = numpy_array_to_live_points(
        np.array([[0.5, 1.0], [2.0, 1.0]]), ["x", "y"]
    )
    model.names = ["x", "y"]
    model.bounds = {"x": [0, 1], "y": [-np.inf, np.inf]}
    val = Model.in_bounds(model, x)
    np.testing.assert_array_equal(val, np.array([True, False]))


def test_parameter_in_bounds(model):
    """Test parameter in bounds method."""
    x = np.array([0, 0.5, 1, 3])
    model.names = ["x", "y"]
    model.bounds = {"x": [0, 1], "y": [0, 4]}
    val = Model.parameter_in_bounds(model, x, "x")
    np.testing.assert_array_equal(val, np.array([True, True, True, False]))


def test_sample_parameter(model):
    """Assert an error is raised."""
    with pytest.raises(NotImplementedError) as excinfo:
        Model.sample_parameter(model, "x", n=2)
    assert "User must implement this method!" in str(excinfo.value)


def test_new_point_single(model):
    """Test the new point when asking for 1 point"""
    model._single_new_point = MagicMock()
    Model.new_point(model, N=1)
    model._single_new_point.assert_called_once()


def test_single_new_point(model, rng):
    """Test the method that draw one point within the prior bounds"""
    model.names = ["x", "y"]
    model.bounds = {"x": [-1, 1], "y": [-2, 2]}
    model.lower_bounds = np.array([-1, -2])
    model.upper_bounds = np.array([1, 2])
    model.log_prior = MagicMock(return_value=0)
    model.dims = 2
    model.rng = rng
    x = Model._single_new_point(model)
    assert (x["x"] >= -1) & (x["x"] <= 1)
    assert (x["y"] >= -2) & (x["y"] <= 2)


def test_new_point_multiple(model):
    """Test the new point when asking for multiple points"""
    model._multiple_new_points = MagicMock()
    Model.new_point(model, N=10)
    model._multiple_new_points.assert_called_once_with(10)


def test_multiple_new_points(model, rng):
    """Test the method that draws multiple points within the prior bounds"""
    n = 10
    model.names = ["x", "y"]
    model.bounds = {"x": [-1, 1], "y": [-2, 2]}
    model.lower_bounds = np.array([-1, -2])
    model.upper_bounds = np.array([1, 2])
    model.log_prior = MagicMock(return_value=np.zeros(10))
    model.dims = 2
    model.rng = rng
    x = Model._multiple_new_points(model, N=n)
    assert x.size == n
    assert ((x["x"] >= -1) & (x["x"] <= 1)).all()
    assert ((x["y"] >= -2) & (x["y"] <= 2)).all()


def test_multiple_new_points_reject(model, rng):
    """Assert the correct number of points are returned in some are rejected"""
    n = 3
    model.names = ["x", "y"]
    model.bounds = {"x": [-1, 1], "y": [-2, 2]}
    model.lower_bounds = np.array([-1, -2])
    model.upper_bounds = np.array([1, 2])
    model.log_prior = MagicMock(
        side_effect=2 * [np.array([-np.inf, 0.0, 0.0])]
    )
    model.rng = rng
    model.dims = 2
    x = Model._multiple_new_points(model, N=n)
    assert x.size == n
    assert ((x["x"] >= -1) & (x["x"] <= 1)).all()
    assert ((x["y"] >= -2) & (x["y"] <= 2)).all()


def test_multiple_new_points_reject_batch(model, rng):
    """Assert rejecting an entire batch does not raise an error"""
    n = 3
    model.names = ["x", "y"]
    model.bounds = {"x": [-1, 1], "y": [-2, 2]}
    model.lower_bounds = np.array([-1, -2])
    model.upper_bounds = np.array([1, 2])
    model.log_prior = MagicMock(
        side_effect=[
            -np.inf * np.ones(3),
            np.zeros(3),
        ]
    )
    model.dims = 2
    model.rng = rng
    x = Model._multiple_new_points(model, N=n)
    assert x.size == n
    assert ((x["x"] >= -1) & (x["x"] <= 1)).all()
    assert ((x["y"] >= -2) & (x["y"] <= 2)).all()


def test_new_point_log_prob(model):
    """Test the log prob for new points.

    Should be zero.
    """
    x = numpy_array_to_live_points(np.random.randn(2, 1), ["x"])
    log_prob = Model.new_point_log_prob(model, x)
    assert log_prob.size == 2
    assert (log_prob == 0).all()


@pytest.mark.integration_test
def test_new_point_integration(integration_model, rng):
    """
    Test the default method for generating a new point with the bounds.

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    integration_model.set_rng(rng)
    new_point = integration_model.new_point()
    log_q = integration_model.new_point_log_prob(new_point)
    assert (new_point["x_0"] < 5) & (new_point["x_0"] > -5)
    assert (new_point["x_1"] < 5) & (new_point["x_1"] > -5)
    assert log_q == 0


@pytest.mark.integration_test
def test_new_point_multiple_integration(integration_model, rng):
    """
    Test drawing multiple new points from the model

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x_0 and x_1.
    """
    integration_model.set_rng(rng)
    new_points = integration_model.new_point(N=100)
    log_q = integration_model.new_point_log_prob(new_points)
    assert new_points.size == 100
    assert all(np.isnan(new_points["logP"]))
    assert all(new_points["x_0"] < 5) & all(new_points["x_1"] > -5)
    assert all(new_points["x_1"] < 5) & all(new_points["x_0"] > -5)
    assert (log_q == 0).all()


def test_likelihood_evaluations(model, live_point):
    """
    Test `evaluate_log_likelihood` and ensure the counter increases.
    """
    model.likelihood_evaluations = 1
    model.log_likelihood = MagicMock(return_value=2)
    log_l = Model.evaluate_log_likelihood(model, live_point)

    model.log_likelihood.assert_called_once_with(live_point)
    assert log_l == 2
    assert model.likelihood_evaluations == 2


def test_likelihood_evaluations_vectorised(model, live_points):
    """
    Test `evaluate_log_likelihood` and ensure the counter increases.
    """
    out = np.random.randn(live_points.size)
    model.likelihood_evaluations = 1
    model.log_likelihood = MagicMock(return_value=out)
    log_l = Model.evaluate_log_likelihood(model, live_points)

    model.log_likelihood.assert_called_once_with(live_points)
    assert log_l is out
    assert model.likelihood_evaluations == (1 + live_points.size)


def test_log_prior(model):
    """Verify the log prior raises a NotImplementedError"""
    with pytest.raises(NotImplementedError):
        Model.log_prior(model, 1)


def test_log_likelihood(model):
    """Verify the log likelihood raises a NotImplementedError"""
    with pytest.raises(NotImplementedError):
        Model.log_likelihood(model, 1)


def test_to_unit_hypercube(model):
    """Assert an error is raised by default"""
    with pytest.raises(NotImplementedError):
        Model.to_unit_hypercube(model, 1)


def test_from_unit_hypercube(model):
    """Assert an error is raised by default"""
    with pytest.raises(NotImplementedError):
        Model.from_unit_hypercube(model, 1)


def test_log_prior_unit_hypercube(model):
    model.names = ["x", "y"]
    x = np.array(
        [(0.5, 0.5), (-0.1, 0.5)], dtype=[(n, "f8") for n in model.names]
    )
    model.unstructured_view = rfn.structured_to_unstructured
    out = Model.log_prior_unit_hypercube(model, x)
    assert out[0] == 0
    assert out[1] == -np.inf


def test_missing_log_prior():
    """
    Test to ensure a model cannot be instantiated without a log-prior method.
    """

    class TestModel(Model):
        def __init__(self):
            self.bounds = {"x": [-5, 5], "y": [-5, 5]}
            self.names = ["x", "y"]

        def log_likelihood(self, x):
            return x

    with pytest.raises(TypeError) as excinfo:
        TestModel()
    assert "Can't instantiate abstract class TestModel" in str(excinfo.value)


def test_missing_log_likelihood():
    """
    Test to ensure a model cannot be instantiated without a log-likelihood
    method.
    """

    class TestModel(Model):
        def __init__(self):
            self.bounds = {"x": [-5, 5], "y": [-5, 5]}
            self.names = ["x", "y"]

        def log_prior(self, x):
            return 0.0

    with pytest.raises(TypeError) as excinfo:
        TestModel()
    assert "Can't instantiate abstract class TestModel" in str(excinfo.value)


def test_model_1d_error():
    """Assert an error is raised if the model is one dimensional."""

    class TestModel(EmptyModel):
        def __init__(self):
            self.names = ["x"]
            self.bounds = {"x": [-5, 5]}

    with pytest.raises(OneDimensionalModelError) as excinfo:
        TestModel()
    assert "nessai is not designed to handle one-dimensional models" in str(
        excinfo.value
    )


def test_verify_new_point(rng):
    """
    Test `Model.verify_model` and ensure a model with an ill-defined
    prior function raises the correct error
    """

    class BrokenModel(BasicModel):
        def log_prior(self, x):
            return -np.inf

    model = BrokenModel()
    model.set_rng(rng)

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert "valid point" in str(excinfo.value)


@pytest.mark.parametrize("log_p", [np.inf, -np.inf])
def test_verify_log_prior_finite(log_p, rng):
    """
    Test `Model.verify_model` and ensure a model with a log-prior that
    only returns inf function raises the correct error
    """

    class BrokenModel(BasicModel):
        def log_prior(self, x):
            return log_p

    model = BrokenModel()
    model.rng = rng

    with pytest.raises(RuntimeError):
        model.verify_model()


def test_verify_log_prior_none(rng):
    """
    Test `Model.verify_model` and ensure a model with a log-prior that
    only returns None raises an error.
    """

    class BrokenModel(BasicModel):
        def log_prior(self, x):
            return None

    model = BrokenModel()
    model.set_rng(rng)

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert "Log-prior" in str(excinfo.value)


def test_verify_log_likelihood_none(rng):
    """
    Test `Model.verify_model` and ensure a model with a log-likelihood that
    only returns None raises an error.
    """

    class BrokenModel(BasicModel):
        def log_likelihood(self, x):
            return None

    model = BrokenModel()
    model.set_rng(rng)

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert "Log-likelihood" in str(excinfo.value)


def test_verify_no_names():
    """
    Test `Model.verify_model` and ensure a model without names
    function raises the correct error
    """

    class TestModel(EmptyModel):
        def __init__(self):
            self.bounds = {"x": [-5, 5], "y": [-5, 5]}

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()
    assert "`names` is not set" in str(excinfo.value)


def test_verify_empty_names():
    """Assert an error is raised if names evaluates to false."""

    class TestModel(EmptyModel):
        names = []

        def __init__(self):
            self.bounds = {"x": [-2, 2], "y": [-2, 2]}

    model = TestModel()
    with pytest.raises(ValueError) as excinfo:
        model.verify_model()

    assert "`names` is not set to a valid value" in str(excinfo.value)


def test_verify_invalid_names_type():
    """Assert an error is raised if names is not a list."""

    class TestModel(EmptyModel):
        names = "x"

        def __init__(self):
            self.bounds = {"x": [-2, 2], "y": [-2, 2]}

    model = TestModel()
    with pytest.raises(TypeError) as excinfo:
        model.verify_model()

    assert "`names` must be a list" in str(excinfo.value)


def test_verify_no_bounds():
    """
    Test `Model.verify_model` and ensure a model without bounds
    function raises the correct error
    """

    class TestModel(EmptyModel):
        def __init__(self):
            self.names = ["x", "y"]

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert "`bounds` is not set" in str(excinfo.value)


def test_verify_empty_bounds():
    """Assert an error is raised if bounds evaluates to false."""

    class TestModel(EmptyModel):
        bounds = {}

        def __init__(self):
            self.names = ["x", "y"]

    model = TestModel()

    with pytest.raises(ValueError) as excinfo:
        model.verify_model()

    assert "`bounds` is not set to a valid value" in str(excinfo.value)


def test_verify_invalid_bounds_type():
    """Assert an error is raised if bounds are not a dictionary."""

    class TestModel(EmptyModel):
        bounds = []

        def __init__(self):
            self.names = ["x", "y"]

    model = TestModel()

    with pytest.raises(TypeError) as excinfo:
        model.verify_model()

    assert "`bounds` must be a dictionary" in str(excinfo.value)


def test_verify_incomplete_bounds(rng):
    """
    Test `Model.verify_model` and ensure a model without bounds
    function raises the correct error
    """

    class TestModel(EmptyModel):
        bounds = {"x": [-5, 5]}

        def __init__(self):
            self.names = ["x", "y"]

    model = TestModel()
    model.rng = rng

    with pytest.raises(RuntimeError):
        model.verify_model()


def test_verify_model_1d(rng):
    """Assert an error is raised if the model is one dimensional."""

    class TestModel(EmptyModel):
        names = ["x"]
        bounds = {"x": [-5, 5]}

    model = TestModel()
    model.set_rng(rng)

    with pytest.raises(OneDimensionalModelError) as excinfo:
        model.verify_model()
    assert "nessai is not designed to handle one-dimensional models" in str(
        excinfo.value
    )


@pytest.mark.parametrize("value", [np.log(True), np.float16(5.0)])
def test_verify_float16(caplog, value, rng):
    """
    Test `Model.verify_model` and ensure that a critical warning is raised
    if a float16 array is returned by the prior.
    """

    class BrokenModel(BasicModel):
        def log_prior(self, x):
            return value

    model = BrokenModel()
    model.set_rng(rng)

    model.verify_model()

    assert "float16 precision" in caplog.text


def test_verify_no_float16(caplog, rng):
    """
    Test `Model.verify_model` and ensure that a critical warning is not raised
    if array return by log_prior is not dtype float16.
    """
    model = BasicModel()
    model.set_rng(rng)
    out = model.verify_model()
    assert out is True
    assert "float16 precision" not in caplog.text


def test_unbounded_priors_wo_new_point(rng):
    """Test `Model.verify_model` with unbounded priors"""

    class TestModel(Model):
        def __init__(self):
            self.names = ["x", "y"]
            self.bounds = {"x": [-5, 5], "y": [-np.inf, np.inf]}

        def log_prior(self, x):
            return -np.log(10) * np.ones(x.size) + norm.logpdf(x["y"])

        def log_likelihood(self, x):
            return np.ones(x.size)

    model = TestModel()
    model.set_rng(rng)
    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert "Could not draw a new point" in str(excinfo.value)


def test_unbounded_priors_w_new_point(rng):
    """Test `Model.verify_model` with unbounded priors"""

    class TestModel(Model):
        def __init__(self):
            self.names = ["x", "y"]
            self.bounds = {"x": [-5, 5], "y": [-np.inf, np.inf]}

        def new_point(self, N=1):
            return numpy_array_to_live_points(
                np.concatenate(
                    [np.random.uniform(-5, 5, (N, 1)), norm.rvs(size=(N, 1))],
                    axis=1,
                ),
                self.names,
            )

        def log_prior(self, x):
            return -np.log(10) * np.ones(x.size) + norm.logpdf(x["y"])

        def log_likelihood(self, x):
            return np.ones(x.size)

    model = TestModel()
    model.set_rng(rng)
    model.verify_model()


def test_verify_model_likelihood_repeated_calls(rng):
    """Assert that an error is raised if repeated calls with the likelihood
    return different values.
    """

    class BrokenModel(BasicModel):
        count = 0
        allow_multi_valued_likelihood = False

        def log_likelihood(self, x):
            self.count += 1
            return self.count

    model = BrokenModel()
    model.set_rng(rng)

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()
    assert "Repeated calls" in str(excinfo.value)


def test_verify_model_likelihood_repeated_calls_allowed(caplog, rng):
    """Assert allow multi valued likelihood prevents an error from being
    raised.
    """

    class MultiValuedModel(BasicModel):
        allow_multi_valued_likelihood = True

        def log_likelihood(self, x):
            return np.random.rand()

    model = MultiValuedModel()
    model.set_rng(rng)
    model.verify_model()
    assert "Multi-valued likelihood is allowed." in str(caplog.text)


def test_configure_pool_with_pool(model):
    """Test configuring the pool when pool is specified"""
    n_pool = 2
    pool = MagicMock()
    with patch("nessai.model.get_n_pool", return_value=n_pool) as mock:
        Model.configure_pool(model, pool=pool, n_pool=1)
    mock.assert_called_once_with(pool)
    assert model.pool is pool
    assert model.n_pool == n_pool


def test_configure_pool_with_pool_no_n_pool(model):
    """Test configuring the pool when pool is specified but n_pool cannot be
    determined and the user has not specified the value.
    """
    pool = MagicMock()
    with patch("nessai.model.get_n_pool", return_value=None) as mock:
        Model.configure_pool(model, pool=pool)
    mock.assert_called_once_with(pool)
    assert model.pool is pool
    assert model.n_pool is None
    assert model.allow_vectorised is False


def test_configure_pool_with_pool_user_n_pool(model):
    """Test configuring the pool when pool is specified but n_pool cannot be
    determined but the user has specified the value.
    """
    model.allow_vectorised = True
    pool = MagicMock()
    with patch("nessai.model.get_n_pool", return_value=None) as mock:
        Model.configure_pool(model, pool=pool, n_pool=1)
    mock.assert_called_once_with(pool)
    assert model.pool is pool
    assert model.n_pool == 1


def test_configure_pool_n_pool(model):
    """Test configuring the pool when n_pool is specified"""
    n_pool = 1
    pool = MagicMock()
    with (
        patch("multiprocessing.Pool", return_value=pool) as mock_pool,
        patch(
            "nessai.utils.multiprocessing.check_multiprocessing_start_method"
        ) as mock_check,
    ):
        Model.configure_pool(model, n_pool=n_pool)
    assert model.pool is pool

    mock_check.assert_called_once()
    mock_pool.assert_called_once_with(
        processes=n_pool,
        initializer=initialise_pool_variables,
        initargs=(model,),
    )


def test_configure_pool_none(model, caplog):
    """Test configuring the pool when pool and n_pool are None"""
    caplog.set_level(logging.INFO)
    Model.configure_pool(model, pool=None, n_pool=None)
    assert model.pool is None
    assert "pool and n_pool are none, no multiprocessing pool" in str(
        caplog.text
    )


def test_configure_pool_already_configured(model, caplog):
    """Assert configuration is skipped if the pool is already configured"""
    model._pool_configured = True
    model.n_pool = 2
    Model.configure_pool(model, n_pool=4)
    assert model.n_pool == 2
    assert "pool has already been configured" in str(caplog.text)


@pytest.mark.parametrize("code", [10, 2])
def test_close_pool(model, code):
    """Test closing the pool"""
    pool = MagicMock()
    pool.close = MagicMock()
    pool.terminate = MagicMock()
    pool.join = MagicMock()
    model.pool = pool
    Model.close_pool(model, code=code)
    pool.join.assert_called_once()
    if code == 2:
        pool.terminate.assert_called_once()
        pool.pool.assert_not_called()
    else:
        pool.close.assert_called_once()
        pool.terminate.assert_not_called()
    assert model.pool is None


@pytest.mark.parametrize("chunksize", [None, 3])
def test_evaluate_likelihoods_pool_vectorised(model, chunksize):
    """Test evaluating a vectorised likelihood with a pool."""
    samples = numpy_array_to_live_points(
        np.array([1, 2, 3, 4])[:, np.newaxis], ["x"]
    )
    logL = [np.array([-1, -2]), np.array([-3, -4])]
    expected = np.array([-1, -2, -3, -4])
    model.pool = MagicMock(side_effect=True)
    model.n_pool = 2
    model.vectorised_likelihood = True
    model.allow_vectorised = True
    model.likelihood_chunksize = chunksize
    model.pool.map = MagicMock(return_value=logL)
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100

    out = Model.batch_evaluate_log_likelihood(model, samples)

    assert model.pool.map.call_args_list[0][0][0] is log_likelihood_wrapper
    input_array = model.pool.map.call_args_list[0][0][1]
    if chunksize is None:
        assert len(input_array) == 2
        assert_structured_arrays_equal(
            input_array, np.array([samples[:2], samples[2:]])
        )
    else:
        assert len(input_array) == 2
        for i, a in enumerate(input_array):
            assert_structured_arrays_equal(
                a, samples[i * chunksize : (i + 1) * chunksize]
            )

    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 104
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("chunksize", [None, 1])
def test_evaluate_likelihoods_pool_not_vectorised(model, chunksize):
    """Test evaluating the likelihood with a pool"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ["x"])
    logL = np.array([3, 4])
    model.pool = MagicMock(side_effect=True)
    model.n_pool = 2
    model.vectorised_likelihood = False
    model.allow_vectorised = True
    model.likelihood_chunksize = chunksize
    model.pool.map = MagicMock(return_value=logL)
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.pool.map.assert_called_once_with(log_likelihood_wrapper, samples)
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_evaluate_likelihoods_no_pool_not_vectorised(model):
    """Test evaluating the likelihood without a pool"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ["x"])
    # Cannot compare NaNs in has calls
    samples["logL"] = 0.0
    samples["logP"] = 0.0
    logL = np.array([3, 4])
    model.pool = None
    model.n_pool = None
    model.vectorised_likelihood = False
    model.allow_vectorised = True
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    model.log_likelihood = MagicMock(side_effect=logL)
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.log_likelihood.assert_has_calls([call(samples[0]), call(samples[1])])
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_evaluate_likelihoods_no_pool_vectorised(model):
    """
    Test evaluating the likelihood without a pool but with a vectorised
    likelihood.
    """
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ["x"])
    logL = np.array([3, 4])
    model.pool = None
    model.n_pool = None
    model.vectorised_likelihood = True
    model.allow_vectorised = True
    model.likelihood_chunksize = None
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    model.log_likelihood = MagicMock(return_value=logL)
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.log_likelihood.assert_called_once_with(samples)
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


@pytest.mark.parametrize("chunksize", [10, 12])
def test_evaluate_likelihood_vectorised_chunksize(model, chunksize):
    """Assert the likelihood is called the correct number of times"""
    n = 100
    n_calls = np.ceil(n / chunksize)
    samples = numpy_array_to_live_points(np.random.rand(n, 1), ["x"])
    model.vectorised_likelihood = True
    model.allow_vectorised = True
    model.pool = None
    model.n_pool = None
    model.likelihood_chunksize = chunksize
    model.log_likelihood = MagicMock(
        side_effect=lambda x: np.random.rand(x.size)
    )
    out = Model.batch_evaluate_log_likelihood(model, samples)
    assert model.log_likelihood.call_count == n_calls
    assert len(out) == n


def test_evaluate_likelihoods_allow_vectorised_false(model):
    """Assert that vectorisation isn't used if allow_vectorised is false"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ["x"])
    logL = [3, 4]
    model.pool = None
    model.n_pool = None
    model.vectorised_likelihood = True
    model.allow_vectorised = False
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    model.log_likelihood = MagicMock(side_effect=logL)
    out = Model.batch_evaluate_log_likelihood(model, samples)
    assert model.log_likelihood.call_count == 2
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_evaluate_likelihoods_pool_allow_vectorised_false(model):
    """Test evaluating the likelihood with a pool and allow_vectorised=False"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ["x"])
    logL = np.array([3, 4])
    model.pool = MagicMock(side_effect=True)
    model.n_pool = 2
    model.vectorised_likelihood = True
    model.allow_vectorised = False
    model.pool.map = MagicMock(return_value=logL)
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.pool.map.assert_called_once_with(log_likelihood_wrapper, samples)
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_view_dtype(model):
    """Assert view dtype calls the correct functions with the correct inputs"""
    model._dtype = None
    model.names = ["x", "y"]

    array = np.empty(
        (0,), dtype=[(n, "f8") for n in model.names + ["logL" + "logP"]]
    )
    expected = np.dtype([(n, "f8") for n in model.names])

    with (
        patch(
            "nessai.model._unstructured_view_dtype", return_value=expected
        ) as mock,
        patch(
            "nessai.model.empty_structured_array", return_value=array
        ) as empty_mock,
    ):
        dtype = Model._view_dtype.__get__(model)

    empty_mock.assert_called_once_with(0, model.names)
    mock.assert_called_once_with(array, model.names)
    assert model._dtype == expected
    assert dtype == expected


def test_unstructured_view(model, live_points, rng):
    """Assert the underlying function is called with the correct inputs"""
    out = rng.standard_normal((live_points.size, 2))
    dtype = np.dtype([("x", "f8"), ("y", "f8")])
    model._view_dtype = dtype
    with patch("nessai.model.unstructured_view", return_value=out) as mock:
        view = Model.unstructured_view(model, live_points)

    mock.assert_called_once_with(live_points, dtype=dtype)
    assert view is out


def test_get_state(model):
    """Assert pool is removed in __getstate__"""
    pool = True
    model.pool = pool
    d = Model.__getstate__(model)
    assert d["pool"] is None
    assert model.pool is pool


@pytest.mark.integration_test
@pytest.mark.parametrize("pickleable", [False, True])
@pytest.mark.parametrize("init", ["before", "after", "function"])
def test_pool(integration_model, mp_context, pickleable, init, rng):
    """Integration test for evaluating the likelihood with a pool"""
    integration_model.rng = rng
    method = mp_context.get_start_method()

    if not pickleable:
        # Cannot pickle lambda functions
        integration_model.fn = lambda x: x
        if method != "fork":
            pytest.xfail(f"start method {method} requires a pickleable model")

    if init == "before":
        if method != "fork":
            pytest.xfail(f"Must use initializer in Pool with {method}")
        initialise_pool_variables(integration_model)
        pool = mp_context.Pool(1)
    elif init == "function":
        pool = mp_context.Pool(
            1,
            initializer=initialise_pool_variables,
            initargs=(integration_model,),
        )
    elif init == "after":
        if method != "fork":
            pytest.xfail(f"Must use initializer in Pool with {method}")
        pool = mp_context.Pool(1)
        initialise_pool_variables(integration_model)
    else:
        raise ValueError(init)

    integration_model.configure_pool(pool=pool)
    assert integration_model.pool is pool
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.array(
        [integration_model.log_likelihood(xx) for xx in x],
        dtype=config.livepoints.logl_dtype,
    ).flatten()
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None


@pytest.mark.requires("ray")
@pytest.mark.integration_test
@pytest.mark.flaky(reruns=3)
def test_pool_ray(integration_model, rng):
    """Integration test for evaluating the likelihood with a pool from ray.

    This will break if the class for integration_model is defined globally.
    """
    from ray.util.multiprocessing import Pool

    integration_model.set_rng(rng)

    # Cannot pickle lambda functions
    integration_model.fn = lambda x: x
    pool = Pool(
        processes=1,
        initializer=initialise_pool_variables,
        initargs=(integration_model,),
    )
    integration_model.configure_pool(pool=pool)
    assert integration_model.pool is pool
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.array(
        [integration_model.log_likelihood(xx) for xx in x],
        dtype=config.livepoints.logl_dtype,
    ).flatten()
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None


@pytest.mark.requires("multiprocess")
def test_pool_multiprocess(integration_model, rng):
    """Integration test for evaluating the likelihood with a pool from
    multiprocess.
    """
    import multiprocess as mp

    integration_model.rng = rng
    integration_model.fn = lambda x: x
    pool = mp.Pool(
        1,
        initializer=initialise_pool_variables,
        initargs=(integration_model,),
    )

    integration_model.configure_pool(pool=pool)
    assert integration_model.pool is pool
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.array(
        [integration_model.log_likelihood(xx) for xx in x],
        dtype=config.livepoints.logl_dtype,
    ).flatten()
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None


@pytest.mark.integration_test
@pytest.mark.parametrize("pickleable", [False, True])
def test_n_pool(integration_model, mp_context, pickleable, rng):
    """Integration test for evaluating the likelihood with n_pool"""
    integration_model.set_rng(rng)
    if not pickleable:
        # Cannot pickle lambda functions
        integration_model.fn = lambda x: x
        method = mp_context.get_start_method()
        if method != "fork":
            pytest.xfail(f"start method {method} requires a pickleable model")

    with (
        patch("multiprocessing.Pool", mp_context.Pool),
        patch(
            "nessai.utils.multiprocessing.multiprocessing.get_start_method",
            mp_context.get_start_method,
        ),
    ):
        integration_model.configure_pool(n_pool=1)
    assert integration_model.n_pool == 1
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.array(
        [integration_model.log_likelihood(xx) for xx in x],
        dtype=config.livepoints.logl_dtype,
    ).flatten()
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None


@pytest.mark.integration_test
def test_unstructured_view_integration(integration_model, live_points):
    """Integration test for unstructured view."""
    view = integration_model.unstructured_view(live_points)
    assert view.base is live_points
    assert view.shape == (live_points.size, integration_model.dims)


@pytest.mark.integration_test
def test_in_bounds_integration_values(integration_model, rng):
    """Assert the correct booleans are returned"""
    integration_model.set_rng(rng)
    x = integration_model.new_point(3)
    names = integration_model.names
    x[names[0]][0] = integration_model.bounds[names[0]][1] + 1.0
    x[names[1]][1] = integration_model.bounds[names[1]][0] - 1.0

    expected = np.array([False, False, True])
    out = integration_model.in_bounds(x)
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.integration_test
def test_in_bounds_integration_n_samples(integration_model, n, rng):
    """Assert single and multiple samples work"""
    integration_model.set_rng(rng)
    x = numpy_array_to_live_points(
        rng.standard_normal((n, integration_model.dims)),
        integration_model.names,
    )
    flags = integration_model.in_bounds(x)
    assert len(flags) == n


def test_rng_not_set_single(model):
    model.rng = None
    with pytest.raises(RNGNotSetError):
        Model._single_new_point(model)


def test_rng_not_set_multiple(model):
    model.rng = None
    with pytest.raises(RNGNotSetError):
        Model._multiple_new_points(model, 10)


def test_rng_not_set_verify_model(model):
    model.rng = None
    model.names = ["x", "y"]
    model.bounds = {"x": [-1, 1], "y": [-2, 2]}
    with pytest.raises(RNGNotSetError):
        Model.verify_model(model)


def test_rng_not_set_sample_unit_hypercube(model):
    model.rng = None
    with pytest.raises(RNGNotSetError):
        Model.sample_unit_hypercube(model)


def test_set_rng(model, rng):
    model.rng = None
    Model.set_rng(model, rng)
    assert model.rng is rng


def test_set_rng_not_specified(model, rng):
    model.rng = None
    with patch(
        "numpy.random.default_rng", return_value=rng
    ) as mock_default_rng:
        Model.set_rng(model)
    mock_default_rng.assert_called_once()
    assert model.rng is rng


def test_set_rng_already_set(model, rng):
    model.rng = rng
    with pytest.raises(RNGSetError):
        Model.set_rng(model, rng)
