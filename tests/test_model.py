import numpy as np
import pytest

from flowproposal.model import Model


def test_no_names(model):
    """
    Test the behaviour of the model if the names list is empty
    """
    from flowproposal.model import Model
    m = Model()
    assert m.dims is None


def test_new_point(model):
    """
    Test the default method for generating a new point with the bounds.

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    new_point = model.new_point()
    assert (new_point['x'] < 5) & (new_point['y'] > -5)
    assert (new_point['y'] < 5) & (new_point['y'] > -5)


def test_new_point_multiple(model):
    """
    Test drawing multiple new points from the model

    Uses the model defined in `conftest.py` with bounds [-5, 5] for
    x and y.
    """
    new_points = model.new_point(N=100)
    assert new_points.size == 100
    assert all(np.isfinite(new_points['logP']))
    assert all(new_points['x'] < 5) & all(new_points['y'] > -5)
    assert all(new_points['y'] < 5) & all(new_points['y'] > -5)


def test_verify_new_pooint():
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
