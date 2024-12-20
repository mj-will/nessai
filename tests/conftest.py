"""General configuration for the test suite"""

import logging
import multiprocessing
import sys
import time

import numpy as np
import pytest
import torch
from scipy.stats import norm

from nessai.livepoint import (
    add_extra_parameters_to_live_points,
    reset_extra_live_points_parameters,
)
from nessai.model import Model
from nessai.utils.testing import IntegrationTestModel

torch.manual_seed(170817)

_requires_dependency_cache = dict()


@pytest.fixture()
def rng():
    return np.random.default_rng(170817)


@pytest.fixture()
def model(rng):
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

        def to_unit_hypercube(self, x):
            x_out = x.copy()
            for n in self.names:
                x_out[n] = (x[n] - self.bounds[n][0]) / np.ptp(self.bounds[n])
            return x_out

        def from_unit_hypercube(self, x):
            x_out = x.copy()
            for n in self.names:
                x_out[n] = np.ptp(self.bounds[n]) * x[n] + self.bounds[n][0]
            return x_out

    m = TestModel()
    m.set_rng(rng)
    return m


@pytest.fixture(scope="function")
def integration_model(rng):
    model = IntegrationTestModel()
    return model


@pytest.fixture()
def flow_config():
    return dict(
        n_blocks=2,
        n_neurons=2,
        n_layers=1,
        batch_norm_between_layers=False,
    )


@pytest.fixture()
def training_config():
    return dict(
        max_epochs=5,
        device_tag="cpu",
    )


@pytest.fixture()
def wait():
    """Sleep for 0.01s for timing tests.

    Time resolution on Windows is less than Linux and macOS
    """

    def func(*args, **kwargs):
        time.sleep(0.01)

    return func


@pytest.fixture(params=["fork", "forkserver", "spawn"])
def mp_context(request):
    """Multiprocessing context to test"""
    if sys.platform == "win32" and request.param != "spawn":
        pytest.skip("Windows only supports the 'spawn' start method")
    return multiprocessing.get_context(request.param)


@pytest.fixture()
def ins_parameters():
    """Add (and remove) the standard INS parameters for the tests."""
    # Before every test
    add_extra_parameters_to_live_points(["logQ", "logW", "logU"])
    yield
    reset_extra_live_points_parameters()


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
    config.addinivalue_line(
        "markers",
        "reset_logger: reset the logger after the test",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--bilby-compatibility",
        action="store_true",
        default=False,
        help="Run bilby compatibility tests",
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


def pytest_runtest_teardown(item):
    if "reset_logger" in item.keywords:
        logger = logging.getLogger("nessai")
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
        logger.disabled = False
        logger.filters.clear()
        handlers = logger.handlers.copy()
        for handler in handlers:
            # Copied from `logging.shutdown`.
            try:
                handler.acquire()
                handler.flush()
                handler.close()
            except (OSError, ValueError):
                pass
            finally:
                handler.release()
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())
