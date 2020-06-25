import pytest
import numpy as np
from flowproposal.model import Model


@pytest.fixture()
def model():
    class TestModel(Model):

        def __init__(self):
            self.bounds = np.array([[-5, 5], [-5, 5]])
            self.dims = 2
            self.names = ['x', 'y']

        def log_prior(self, x):
            log_p = 0.
            for i, n in enumerate(self.names):
                log_p += ((x[n] >= self.bounds[i][0]) & (x[n] <= self.bounds[i][1])) \
                        / (self.bounds[i][1] - self.bounds[i][0])
            return log_p
    return TestModel()


def test_new_point(model):
    """
    Test the default method for generating a new point with the bounds
    """
    new_point = model.new_point()
    assert (new_point['x'] < 5) & (new_point['y'] > -5)
    assert (new_point['y'] < 5) & (new_point['y'] > -5)
