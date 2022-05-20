# -*- coding: utf-8 -*-
"""
Tests for `nessai.model`
"""
import datetime
import logging
import numpy as np
import pytest
from scipy.stats import norm
from unittest.mock import MagicMock, call, create_autospec, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model, OneDimensionalModelError
from nessai.utils.multiprocessing import (
    initialise_pool_variables,
    log_likelihood_wrapper,
)


class EmptyModel(Model):

    def log_prior(self, x):
        return None

    def log_likelihood(self, x):
        return None


@pytest.fixture()
def integration_model():
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x))
            for n in self.names:
                log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
            return log_p

        def log_likelihood(self, x):
            log_l = 0
            for pn in self.names:
                log_l += norm.logpdf(x[pn])
            return log_l

    return TestModel()


@pytest.fixture
def model():
    return create_autospec(Model)


def test_names(model):
    """Assert names returns the correct value."""
    model._names = ['x', 'y']
    assert Model.names.__get__(model) is model._names


def test_names_setter(model):
    """Assert names is correctly set"""
    Model.names.__set__(model, ['x', 'y'])
    assert model._names == ['x', 'y']


def test_names_invalid_type(model):
    """Assert an error is raised if `names` is not a list."""
    with pytest.raises(TypeError) as excinfo:
        Model.names.__set__(model, True)
    assert '`names` must be a list' in str(excinfo.value)


def test_names_empty_list(model):
    """Assert an error is raised if `names` is empty."""
    with pytest.raises(ValueError) as excinfo:
        Model.names.__set__(model, [])
    assert '`names` list is empty' in str(excinfo.value)


def test_names_1d_list(model):
    """Assert an error is raised if `names` is 1d"""
    with pytest.raises(OneDimensionalModelError) as excinfo:
        Model.names.__set__(model, ['x'])
    assert 'names list has length 1' in str(excinfo.value)


def test_bounds(model):
    """Assert bounds returns the correct value."""
    model._bounds = {'x': [-1, 1], 'y': [-1, 1]}
    assert Model.bounds.__get__(model) is model._bounds


def test_bounds_setter(model):
    """Assert bounds is correctly set"""
    Model.bounds.__set__(model, {'x': [-1, 1], 'y': [-2, 2]})
    assert list(model._bounds.keys()) == ['x', 'y']
    np.testing.assert_array_equal(model._bounds['x'], [-1, 1])
    np.testing.assert_array_equal(model._bounds['y'], [-2, 2])


def test_bounds_invalid_type(model):
    """Assert an error is raised if `bounds` is not a dictionary."""
    with pytest.raises(TypeError) as excinfo:
        Model.bounds.__set__(model, True)
    assert '`bounds` must be a dictionary' in str(excinfo.value)


def test_bounds_1d(model):
    """Assert an error is raised if `bounds` is 1d"""
    with pytest.raises(OneDimensionalModelError) as excinfo:
        Model.bounds.__set__(model, {'x': [0, 1]})
    assert 'bounds dictionary has length 1' in str(excinfo.value)


@pytest.mark.parametrize('b', [[1], [1, 2, 3]])
def test_bounds_incorrect_length(model, b):
    """Assert an error is raised if `bounds` is 1d"""
    with pytest.raises(ValueError) as excinfo:
        Model.bounds.__set__(model, {'x': b, 'y': [0, 1]})
    assert 'Each entry in `bounds` must have length 2' in str(excinfo.value)


def test_dims(model):
    """Ensure dims are correct"""
    model.names = ['x', 'y']
    assert Model.dims.__get__(model) == 2


def test_dims_no_names(model):
    """Test the behaviour dims when names is empty"""
    model.names = []
    assert Model.dims.__get__(model) is None


def test_lower_bounds(model):
    """Check the lower bounds are correctly set"""
    model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    model._lower = None
    assert (Model.lower_bounds.__get__(model) == [-1, -1]).all()


def test_upper_bounds(model):
    """Check the upper bounds are correctly set"""
    model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    model._upper = None
    assert (Model.upper_bounds.__get__(model) == [1, 1]).all()


def test_vectorised_likelihood_true(model):
    """
    Assert the value is True if the likelihood is vectorised and returns
    the same values.
    """

    # Just return the input since this should always work
    def dummy_likelihood(x):
        return x

    model._vectorised_likelihood = None
    model.allow_vectorised = True
    model.log_likelihood = MagicMock(side_effect=dummy_likelihood)
    model.new_point = MagicMock(return_value=np.random.rand(10))

    out = Model.vectorised_likelihood.__get__(model)
    assert model._vectorised_likelihood is True
    assert out is True


def test_vectorised_likelihood_false(model):
    """
    Assert the value is False if allowed_vectorised but the values do not
    agree.
    """

    def dummy_likelihood(x):
        return np.log(np.random.rand(x.size))

    model._vectorised_likelihood = None
    model.allow_vectorised = True
    model.log_likelihood = MagicMock(side_effect=dummy_likelihood)
    model.new_point = MagicMock(return_value=np.random.rand(10))

    out = Model.vectorised_likelihood.__get__(model)
    assert model._vectorised_likelihood is False
    assert out is False


@pytest.mark.parametrize('error', [TypeError, ValueError])
def test_vectorised_likelihood_not_vectorised_error(model, error):
    """
    Assert the value is False if the likelihood is not vectorised and raises
    an error.
    """

    def dummy_likelihood(x):
        if hasattr(x, '__len__'):
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
    Model.vectorised_likelihood.__set__(model, 'test')
    assert model._vectorised_likelihood == 'test'


def test_in_bounds(model):
    """Test the `in_bounds` method.

    Tests both finite and infinite prior bounds.
    """
    x = numpy_array_to_live_points(np.array([[0.5, 1], [2, 1]]), ['x', 'y'])
    model.names = ['x', 'y']
    model.bounds = {'x': [0, 1], 'y': [-np.inf, np.inf]}
    val = Model.in_bounds(model, x)
    np.testing.assert_array_equal(val, np.array([True, False]))


def test_parameter_in_bounds(model):
    """Test parameter in bounds method."""
    x = np.array([0, 0.5, 1, 3])
    model.names = ['x', 'y']
    model.bounds = {'x': [0, 1], 'y': [0, 4]}
    val = Model.parameter_in_bounds(model, x, 'x')
    np.testing.assert_array_equal(val, np.array([True, True, True, False]))


def test_sample_parameter(model):
    """Assert an error is raised."""
    with pytest.raises(NotImplementedError) as excinfo:
        Model.sample_parameter(model, 'x', n=2)
    assert 'User must implement this method!' in str(excinfo.value)


def test_new_point_single(model):
    """Test the new point when asking for 1 point"""
    model._single_new_point = MagicMock()
    Model.new_point(model, N=1)
    model._single_new_point.assert_called_once()


def test_single_new_point(model):
    """Test the method that draw one point within the prior bounds"""
    model.names = ['x', 'y']
    model.bounds = {'x': [-1, 1], 'y': [-2, 2]}
    model.lower_bounds = np.array([-1, -2])
    model.upper_bounds = np.array([1, 2])
    model.log_prior = MagicMock(return_value=0)
    model.dims = 2
    x = Model._single_new_point(model)
    assert ((x['x'] >= -1) & (x['x'] <= 1))
    assert ((x['y'] >= -2) & (x['y'] <= 2))


def test_new_point_multiple(model):
    """Test the new point when asking for multiple points"""
    model._multiple_new_points = MagicMock()
    Model.new_point(model, N=10)
    model._multiple_new_points.assert_called_once_with(10)


def test_multiple_new_points(model):
    """Test the method that draws multiple points within the prior bounds"""
    n = 10
    model.names = ['x', 'y']
    model.bounds = {'x': [-1, 1], 'y': [-2, 2]}
    model.lower_bounds = np.array([-1, -2])
    model.upper_bounds = np.array([1, 2])
    model.log_prior = MagicMock(return_value=np.zeros(10))
    model.dims = 2
    x = Model._multiple_new_points(model, N=n)
    assert x.size == n
    assert ((x['x'] >= -1) & (x['x'] <= 1)).all()
    assert ((x['y'] >= -2) & (x['y'] <= 2)).all()


def test_new_point_log_prob(model):
    """Test the log prob for new points.

    Should be zero.
    """
    x = numpy_array_to_live_points(np.random.randn(2, 1), ['x'])
    log_prob = Model.new_point_log_prob(model, x)
    assert log_prob.size == 2
    assert (log_prob == 0).all()


@pytest.mark.integration_test
def test_new_point_integration(integration_model):
    """
    Test the default method for generating a new point with the bounds.

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    new_point = integration_model.new_point()
    log_q = integration_model.new_point_log_prob(new_point)
    assert (new_point['x'] < 5) & (new_point['y'] > -5)
    assert (new_point['y'] < 5) & (new_point['y'] > -5)
    assert log_q == 0


@pytest.mark.integration_test
def test_new_point_multiple_integration(integration_model):
    """
    Test drawing multiple new points from the model

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    new_points = integration_model.new_point(N=100)
    log_q = integration_model.new_point_log_prob(new_points)
    assert new_points.size == 100
    assert all(np.isfinite(new_points['logP']))
    assert all(new_points['x'] < 5) & all(new_points['x'] > -5)
    assert all(new_points['y'] < 5) & all(new_points['y'] > -5)
    assert (log_q == 0).all()


def test_likelihood_evaluations(model):
    """
    Test `evaluate_log_likelihood` and ensure the counter increases.
    """
    x = 1
    model.likelihood_evaluations = 0
    model.log_likelihood = MagicMock(return_value=2)
    log_l = Model.evaluate_log_likelihood(model, x)

    model.log_likelihood.assert_called_once_with(x)
    assert log_l == 2
    assert model.likelihood_evaluations == 1


def test_log_prior(model):
    """Verify the log prior raises a NotImplementedError"""
    with pytest.raises(NotImplementedError):
        Model.log_prior(model, 1)


def test_log_likelihood(model):
    """Verify the log likelihood raises a NotImplementedError"""
    with pytest.raises(NotImplementedError):
        Model.log_likelihood(model, 1)


def test_missing_log_prior():
    """
    Test to ensure a model cannot be instantiated without a log-prior method.
    """
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

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
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return 0.

    with pytest.raises(TypeError) as excinfo:
        TestModel()
    assert "Can't instantiate abstract class TestModel" in str(excinfo.value)


def test_model_1d_error():
    """Assert an error is raised if the model is one dimensional."""
    class TestModel(EmptyModel):

        def __init__(self):
            self.names = ['x']
            self.bounds = {'x': [-5, 5]}

    with pytest.raises(OneDimensionalModelError) as excinfo:
        TestModel()
    assert 'nessai is not designed to handle one-dimensional models' \
        in str(excinfo.value)


def test_verify_new_point():
    """
    Test `Model.verify_model` and ensure a model with an ill-defined
    prior function raises the correct error
    """
    class TestModel(EmptyModel):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return -np.inf

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert 'valid point' in str(excinfo.value)


@pytest.mark.parametrize("log_p", [np.inf, -np.inf])
def test_verify_log_prior_finite(log_p):
    """
    Test `Model.verify_model` and ensure a model with a log-prior that
    only returns inf function raises the correct error
    """
    class TestModel(EmptyModel):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return log_p

    model = TestModel()

    with pytest.raises(RuntimeError):
        model.verify_model()


def test_verify_log_prior_none():
    """
    Test `Model.verify_model` and ensure a model with a log-prior that
    only returns None raises an error.
    """
    class TestModel(EmptyModel):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return None

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert 'Log-prior' in str(excinfo.value)


def test_verify_log_likelihood_none():
    """
    Test `Model.verify_model` and ensure a model with a log-likelihood that
    only returns None raises an error.
    """
    class TestModel(EmptyModel):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return 0

        def log_likelihood(self, x):
            return None

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert 'Log-likelihood' in str(excinfo.value)


def test_verify_no_names():
    """
    Test `Model.verify_model` and ensure a model without names
    function raises the correct error
    """
    class TestModel(EmptyModel):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()
    assert '`names` is not set' in str(excinfo.value)


def test_verify_empty_names():
    """Assert an error is raised if names evaluates to false."""
    class TestModel(EmptyModel):
        names = []

        def __init__(self):
            self.bounds = {'x': [-2, 2], 'y': [-2, 2]}

    model = TestModel()
    with pytest.raises(ValueError) as excinfo:
        model.verify_model()

    assert '`names` is not set to a valid value' in str(excinfo.value)


def test_verify_invalid_names_type():
    """Assert an error is raised if names is not a list."""
    class TestModel(EmptyModel):
        names = 'x'

        def __init__(self):
            self.bounds = {'x': [-2, 2], 'y': [-2, 2]}

    model = TestModel()
    with pytest.raises(TypeError) as excinfo:
        model.verify_model()

    assert '`names` must be a list' in str(excinfo.value)


def test_verify_no_bounds():
    """
    Test `Model.verify_model` and ensure a model without bounds
    function raises the correct error
    """
    class TestModel(EmptyModel):

        def __init__(self):
            self.names = ['x', 'y']

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert '`bounds` is not set' in str(excinfo.value)


def test_verify_empty_bounds():
    """Assert an error is raised if bounds evaluates to false."""
    class TestModel(EmptyModel):
        bounds = {}

        def __init__(self):
            self.names = ['x', 'y']

    model = TestModel()

    with pytest.raises(ValueError) as excinfo:
        model.verify_model()

    assert '`bounds` is not set to a valid value' in str(excinfo.value)


def test_verify_invalid_bounds_type():
    """Assert an error is raised if bounds are not a dictionary."""
    class TestModel(EmptyModel):
        bounds = []

        def __init__(self):
            self.names = ['x', 'y']

    model = TestModel()

    with pytest.raises(TypeError) as excinfo:
        model.verify_model()

    assert '`bounds` must be a dictionary' in str(excinfo.value)


def test_verify_incomplete_bounds():
    """
    Test `Model.verify_model` and ensure a model without bounds
    function raises the correct error
    """
    class TestModel(EmptyModel):
        bounds = {'x': [-5, 5]}

        def __init__(self):
            self.names = ['x', 'y']

    model = TestModel()

    with pytest.raises(RuntimeError):
        model.verify_model()


def test_verify_model_1d():
    """Assert an error is raised if the model is one dimensional."""
    class TestModel(EmptyModel):
        names = ['x']
        bounds = {'x': [-5, 5]}

    model = TestModel()

    with pytest.raises(OneDimensionalModelError) as excinfo:
        model.verify_model()
    assert 'nessai is not designed to handle one-dimensional models' \
        in str(excinfo.value)


def test_unbounded_priors_wo_new_point():
    """Test `Model.verify_model` with unbounded priors"""

    class TestModel(Model):

        def __init__(self):
            self.names = ['x', 'y']
            self.bounds = {'x': [-5, 5], 'y': [-np.inf, np.inf]}

        def log_prior(self, x):
            return -np.log(10) * np.ones(x.size) + norm.logpdf(x['y'])

        def log_likelihood(self, x):
            return np.ones(x.size)

    model = TestModel()
    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert 'Could not draw a new point' in str(excinfo.value)


def test_unbounded_priors_w_new_point():
    """Test `Model.verify_model` with unbounded priors"""

    class TestModel(Model):

        def __init__(self):
            self.names = ['x', 'y']
            self.bounds = {'x': [-5, 5], 'y': [-np.inf, np.inf]}

        def new_point(self, N=1):
            return numpy_array_to_live_points(
                    np.concatenate([np.random.uniform(-5, 5, (N, 1)),
                                    norm.rvs(size=(N, 1))], axis=1),
                    self.names)

        def log_prior(self, x):
            return -np.log(10) * np.ones(x.size) + norm.logpdf(x['y'])

        def log_likelihood(self, x):
            return np.ones(x.size)

    model = TestModel()
    model.verify_model()


def test_configure_pool_with_pool(model):
    """Test configuring the pool when pool is specified"""
    n_pool = 2
    pool = MagicMock()
    with patch('nessai.model.get_n_pool', return_value=n_pool) as mock:
        Model.configure_pool(model, pool=pool, n_pool=1)
    mock.assert_called_once_with(pool)
    assert model.pool is pool
    assert model.n_pool == n_pool


def test_configure_pool_with_pool_no_n_pool(model):
    """Test configuring the pool when pool is specified but n_pool cannot be
    determined and the user has not specified the value.
    """
    pool = MagicMock()
    with patch('nessai.model.get_n_pool', return_value=None) as mock:
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
    with patch('nessai.model.get_n_pool', return_value=None) as mock:
        Model.configure_pool(model, pool=pool, n_pool=1)
    mock.assert_called_once_with(pool)
    assert model.pool is pool
    assert model.n_pool == 1


def test_configure_pool_n_pool(model):
    """Test configuring the pool when n_pool is specified"""
    n_pool = 1
    pool = MagicMock()
    with patch('multiprocessing.Pool', return_value=pool) as mock_pool:
        Model.configure_pool(model, n_pool=n_pool)
    assert model.pool is pool
    mock_pool.assert_called_once_with(
        processes=n_pool,
        initializer=initialise_pool_variables,
        initargs=(model,)
    )


def test_configure_pool_none(model, caplog):
    """Test configuring the pool when pool and n_pool are None"""
    caplog.set_level(logging.INFO)
    Model.configure_pool(model, pool=None, n_pool=None)
    assert model.pool is None
    assert 'pool and n_pool are none, no multiprocessing pool' \
        in str(caplog.text)


@pytest.mark.parametrize('code', [10, 2])
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


def test_evaluate_likelihoods_pool_vectorised(model):
    """Test evaluating a vectorised likelihood with a pool."""
    samples = numpy_array_to_live_points(
        np.array([1, 2, 3, 4])[:, np.newaxis], ['x']
    )
    logL = [np.array([-1, -2]), np.array([-3, -4])]
    expected = np.array([-1, -2, -3, -4])
    model.pool = MagicMock(side_effect=True)
    model.n_pool = 2
    model.vectorised_likelihood = True
    model.allow_vectorised = True
    model.pool.map = MagicMock(return_value=logL)
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100

    out = Model.batch_evaluate_log_likelihood(model, samples)

    assert model.pool.map.call_args_list[0][0][0] is log_likelihood_wrapper

    input_array = model.pool.map.call_args_list[0][0][1]
    assert len(input_array) == 2
    np.testing.assert_array_equal(
        input_array, np.array([samples[:2], samples[2:]])
    )
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 104
    np.testing.assert_array_equal(out, expected)


def test_evaluate_likelihoods_pool_not_vectorised(model):
    """Test evaluating the likelihood with a pool"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = np.array([3, 4])
    model.pool = MagicMock(side_effect=True)
    model.n_pool = 2
    model.vectorised_likelihood = False
    model.allow_vectorised = True
    model.pool.map = MagicMock(return_value=logL)
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.pool.map.assert_called_once_with(
        log_likelihood_wrapper,
        samples
    )
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_evaluate_likelihoods_no_pool_not_vectorised(model):
    """Test evaluating the likelihood without a pool"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = np.array([3, 4])
    model.pool = None
    model.vectorised_likelihood = False
    model.allow_vectorised = True
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    model.log_likelihood = MagicMock(side_effect=logL)
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.log_likelihood.assert_has_calls(
        [call(samples[0]), call(samples[1])]
    )
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_evaluate_likelihoods_no_pool_vectorised(model):
    """
    Test evaluating the likelihood without a pool but with a vectorised
    likelihood.
    """
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = np.array([3, 4])
    model.pool = None
    model.vectorised_likelihood = True
    model.allow_vectorised = True
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    model.log_likelihood = MagicMock(return_value=logL)
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.log_likelihood.assert_called_once_with(samples)
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_evaluate_likelihoods_allow_vectorised_false(model):
    """Assert that vectorisation isn't used if allow_vectorised is false"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = [3, 4]
    model.pool = None
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
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = np.array([3, 4])
    model.pool = MagicMock(side_effect=True)
    model.n_pool = 2
    model.vectorised_likelihood = True
    model.allow_vectorised = False
    model.pool.map = MagicMock(return_value=logL)
    model.likelihood_evaluation_time = datetime.timedelta()
    model.likelihood_evaluations = 100
    out = Model.batch_evaluate_log_likelihood(model, samples)
    model.pool.map.assert_called_once_with(
        log_likelihood_wrapper,
        samples
    )
    model.likelihood_evaluation_time.total_seconds() > 0
    assert model.likelihood_evaluations == 102
    np.testing.assert_array_equal(out, logL)


def test_get_state(model):
    """Assert pool is removed in __getstate__"""
    pool = True
    model.pool = pool
    d = Model.__getstate__(model)
    assert d['pool'] is None
    assert model.pool is pool


@pytest.mark.integration_test
def test_pool(integration_model):
    """Integration test for evaluating the likelihood with a pool"""
    from multiprocessing import Pool
    # Cannot pickle lambda functions
    integration_model.fn = lambda x: x
    initialise_pool_variables(integration_model)
    pool = Pool(1)
    integration_model.configure_pool(pool=pool)
    assert integration_model.pool is pool
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.fromiter(map(integration_model.log_likelihood, x), 'float')
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None


@pytest.mark.requires('ray')
@pytest.mark.integration_test
def test_pool_ray(integration_model):
    """Integration test for evaluating the likelihood with a pool from ray"""
    from ray.util.multiprocessing import Pool
    # Cannot pickle lambda functions
    integration_model.fn = lambda x: x
    pool = Pool(
        processes=1,
        initializer=initialise_pool_variables,
        initargs=(integration_model,)
    )
    integration_model.configure_pool(pool=pool)
    assert integration_model.pool is pool
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.fromiter(map(integration_model.log_likelihood, x), 'float')
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None


@pytest.mark.integration_test
def test_n_pool(integration_model):
    """Integration test for evaluating the likelihood with n_pool"""
    # Cannot pickle lambda functions
    integration_model.fn = lambda x: x
    integration_model.configure_pool(n_pool=1)
    assert integration_model.n_pool == 1
    x = integration_model.new_point(10)
    out = integration_model.batch_evaluate_log_likelihood(x)

    target = np.fromiter(map(integration_model.log_likelihood, x), 'float')
    np.testing.assert_array_equal(out, target)
    assert integration_model.likelihood_evaluations == 10

    integration_model.close_pool()
    assert integration_model.pool is None
