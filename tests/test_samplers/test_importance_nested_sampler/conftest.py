from unittest.mock import create_autospec

import pytest
from nessai.samplers.importancesampler import ImportanceNestedSampler
from nessai.livepoint import (
    numpy_array_to_live_points,
    add_extra_parameters_to_live_points,
    reset_extra_live_points_parameters,
)
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def ins_livepoint_params():
    reset_extra_live_points_parameters()
    add_extra_parameters_to_live_points(["logW", "logQ"])
    # Test happens here
    yield

    # Called after every test
    reset_extra_live_points_parameters()


@pytest.fixture
def ins():
    return create_autospec(ImportanceNestedSampler)


@pytest.fixture(scope="module")
def n_it():
    return 10


@pytest.fixture
def samples(model, n_it):
    x = numpy_array_to_live_points(np.random.randn(1000, 2), model.names)
    x["it"] = np.random.randint(0, n_it, size=len(x))
    x["logL"] = model.log_likelihood(x)
    x["logP"] = model.log_prior(x)
    return x


@pytest.fixture
def log_q(samples, n_it):
    return np.random.randn(samples.size, n_it)


@pytest.fixture
def history(n_it):
    keys = [
        "logX",
        "logZ",
        "min_logL",
        "max_logL",
        "median_logL",
        "likelihood_evaluations",
        "n_post",
        "logL_threshold",
        "n_added",
        "n_removed",
        "n_live",
        "n_post",
        "live_points_ess",
        "leakage_live_points",
        "leakage_new_points",
        "gradients",
        "samples_entropy",
        "proposal_entropy",
    ]
    d = {k: np.random.randn(n_it) for k in keys}
    d["stopping_criteria"] = dict(
        dZ=np.arange(n_it),
        ratio=np.arange(n_it),
        ess=np.arange(n_it),
        kl=np.arange(n_it),
    )
    return d
