"""General configuration for the test suite"""
import sys

from numpy.random import seed
import numpy as np
import pytest
from scipy.stats import norm
import time
import torch
import multiprocessing

from nessai.model import Model


seed(170817)
torch.manual_seed(170817)

_requires_dependency_cache = dict()


@pytest.fixture()
def model():
    class TestModel(Model):
        def __init__(self):
            self.bounds = {"x": [-5, 5], "y": [-5, 5]}
            self.names = ["x", "y"]

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x), dtype="float")
            for n in self.names:
                log_p -= self.bounds[n][1] - self.bounds[n][0]
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
        model_config=dict(
            n_blocks=2,
            n_neurons=2,
            n_layers=1,
            device_tag="cpu",
            kwargs=dict(batch_norm_between_layers=False),
        ),
    )
    return d


@pytest.fixture()
def wait():
    """Sleep for 0.01s for timing tests.

    Time resolution on Windows is less than Linux and macOS
    """

    def func(*args, **kwargs):
        time.sleep(0.01)

    return func


@pytest.fixture(params=["fork", "spawn"])
def mp_context(request):
    """Multiprocessing context to test"""
    if request.param == "spawn":
        pytest.skip(
            "nessai does not currently support multiprocessing with the "
            "'spawn' method."
        )
    if sys.platform == "win32":
        pytest.skip("Windows does not support 'fork' method")
    return multiprocessing.get_context(request.param)


def pytest_addoption(parser):
    parser.addoption(
        "--bilby-compatibility",
        action="store_true",
        default=False,
        help="Run bilby compatibility tests",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "cuda: mark test to indicate it requires CUDA"
    )
    config.addinivalue_line(
        "markers",
        "requires(package): mark test to only run if the package can be "
        "imported",
    )
    config.addinivalue_line(
        "markers",
        "skip_on_windows: mark test to indicated it should be skipped on "
        "Windows",
    )
    config.addinivalue_line(
        "markers",
        "bilby_compatibility: mark test as a bilby compatibility test, these "
        "tests will be skipped unless the command line option is specified.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--bilby-compatibility"):
        return
    skip_bilby_compat = pytest.mark.skip(
        reason="Need --bilby-compatibility to run"
    )
    for item in items:
        if "bilby_compatibility" in item.keywords:
            item.add_marker(skip_bilby_compat)


def pytest_runtest_setup(item):
    for mark in item.iter_markers(name="cuda"):
        if not torch.cuda.is_available():
            pytest.skip("Test requires CUDA")
    for mark in item.iter_markers(name="requires"):
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

        reason = "Missing dependency: {}".format(name)
        if skip_it:
            pytest.skip(reason)
    for mark in item.iter_markers(name="skip_on_windows"):
        if sys.platform == "win32":
            pytest.skip("Test does not run on Windows")
