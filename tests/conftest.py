
import pytest
from scipy.stats import norm
import torch

from nessai.model import Model

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="test requires CUDA")


@pytest.fixture(scope='session')
def model():
    class TestModel(Model):

        def __init__(self):
            self.bounds = {'x': [-5, 5], 'y': [-5, 5]}
            self.names = ['x', 'y']

        def log_prior(self, x):
            log_p = 0.
            for n in self.names:
                log_p += ((x[n] >= self.bounds[n][0])
                          & (x[n] <= self.bounds[n][1])) \
                        / (self.bounds[n][1] - self.bounds[n][0])
            return log_p

        def log_likelihood(self, x):
            log_l = 0
            for pn in self.names:
                log_l += norm.logpdf(x[pn])
            return log_l

    return TestModel()


@pytest.fixture(scope='session')
def flow_config():
    d = dict(
            max_epochs=5,
            model_config=dict(n_blocks=2, n_neurons=2, n_layers=1,
                              device_tag='cpu',
                              kwargs=dict(batch_norm_between_layers=False))
            )
    return d
