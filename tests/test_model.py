import numpy as np
import pytest

from nessai.model import Model


def test_no_names(model):
    """
    Test the behaviour of the model if the names list is empty
    """
    m = Model()
    assert m.dims is None


def test_new_point(model):
    """
    Test the default method for generating a new point with the bounds.

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    new_point = model.new_point()
    log_q = model.new_point_log_prob(new_point)
    assert (new_point['x'] < 5) & (new_point['y'] > -5)
    assert (new_point['y'] < 5) & (new_point['y'] > -5)
    assert log_q == 0


def test_new_point_multiple(model):
    """
    Test drawing multiple new points from the model

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    new_points = model.new_point(N=100)
    log_q = model.new_point_log_prob(new_points)
    assert new_points.size == 100
    assert all(np.isfinite(new_points['logP']))
    assert all(new_points['x'] < 5) & all(new_points['x'] > -5)
    assert all(new_points['y'] < 5) & all(new_points['y'] > -5)
    assert (log_q == 0).all()


@pytest.mark.parametrize("N", [1, 100])
def test_likelihood_evaluations(model, N):
    """
    Test `evaluate_log_likelihood` and ensure the counter increases.
    """
    n_eval = model.likelihood_evaluations
    new_points = model.new_point(N=N)
    log_l = model.evaluate_log_likelihood(new_points)

    assert log_l.size == N
    assert (model.likelihood_evaluations - n_eval) == N


def test_verify_new_point():
    """
    Test `Model.verify_model` and ensure a model with an ill-defined
    prior function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return -np.inf

    model = TestModel()

    with pytest.raises(RuntimeError) as excinfo:
        model.verify_model()

    assert 'valid point' in str(excinfo.value)


def test_verify_log_prior():
    """
    Test `Model.verify_model` and ensure a model without a log-prior
    function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

    model = TestModel()

    with pytest.raises(RuntimeError):
        model.verify_model()


@pytest.mark.parametrize("log_p", [np.inf, -np.inf])
def test_verify_log_prior_finite(log_p):
    """
    Test `Model.verify_model` and ensure a model with a log-prior that
    only returns inf function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return log_p

    model = TestModel()

    with pytest.raises(RuntimeError):
        model.verify_model()


def test_verify_log_likelihood():
    """
    Test `Model.verify_model` and ensure a model without a log-likelihood
    function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            return 0.

    model = TestModel()

    with pytest.raises(RuntimeError):
        model.verify_model()


def test_verify_no_names():
    """
    Test `Model.verify_model` and ensure a model without names
    function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.names = []
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}

    model = TestModel()

    with pytest.raises(ValueError):
        model.verify_model()


def test_verify_no_bounds():
    """
    Test `Model.verify_model` and ensure a model without bounds
    function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.names = ['x', 'y']
            self.bounds = {}

    model = TestModel()

    with pytest.raises(ValueError) as excinfo:
        model.verify_model()

    assert 'Bounds are not set' in str(excinfo.value)


def test_verify_incomplete_bounds():
    """
    Test `Model.verify_model` and ensure a model without bounds
    function raises the correct error
    """
    class TestModel(Model):

        def __init__(self):
            self.names = ['x', 'y']
            self.bounds = {'x': [-5, 5]}

    model = TestModel()

    with pytest.raises(RuntimeError):
        model.verify_model()
