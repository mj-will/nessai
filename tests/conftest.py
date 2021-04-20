
from numpy.random import seed
import pytest
from scipy.stats import norm
import torch

from nessai.model import Model


seed(170817)
torch.manual_seed(170817)

_requires_dependency_cache = dict()


@pytest.fixture()
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


@pytest.fixture()
def flow_config():
    d = dict(
            max_epochs=5,
            model_config=dict(n_blocks=2, n_neurons=2, n_layers=1,
                              device_tag='cpu',
                              kwargs=dict(batch_norm_between_layers=False))
            )
    return d


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "cuda: mark test to indicate it requires CUDA"
    )
    config.addinivalue_line(
        "markers",
        "requires(package): mark test to only run if the package can be "
        "imported"
    )


def pytest_runtest_setup(item):
    for mark in item.iter_markers(name='cuda'):
        if not torch.cuda.is_available():
            pytest.skip('Test requires CUDA')
    for mark in item.iter_markers(name='requires'):
        name = mark.args[0]
        if name in _requires_dependency_cache:
            skip_it = _requires_dependency_cache[name]
        else:
            try:
                __import__(name)
                skip_it = False
            except ImportError:
                skip_it = True

            _requires_dependency_cache[name] = skip_it

        reason = 'Missing dependency: {}'.format(name)
        if skip_it:
            pytest.skip(reason)
