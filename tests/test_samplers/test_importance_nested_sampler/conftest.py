from unittest.mock import create_autospec

import pytest
from nessai.proposal.importance import ImportanceFlowProposal
from nessai.samplers.importancesampler import ImportanceNestedSampler
from nessai.livepoint import (
    numpy_array_to_live_points,
    add_extra_parameters_to_live_points,
    reset_extra_live_points_parameters,
)
from nessai.model import Model
import numpy as np
from scipy.special import logsumexp


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
    obj = create_autospec(ImportanceNestedSampler)
    obj.model = create_autospec(Model)
    return obj


@pytest.fixture()
def proposal():
    return create_autospec(ImportanceFlowProposal)


@pytest.fixture(scope="module")
def n_it():
    return 10


@pytest.fixture(scope="module")
def n_samples():
    return 1000


@pytest.fixture
def samples(model, n_samples, n_it, log_q):
    x = numpy_array_to_live_points(
        np.random.randn(n_samples, len(model.names)), model.names
    )
    x["it"] = np.random.randint(0, n_it, size=len(x))
    x["logL"] = model.log_likelihood(x)
    x["logP"] = model.log_prior(x)

    alpha = np.unique(x["it"]).astype(float)
    alpha /= alpha.sum()
    x["logQ"] = logsumexp(log_q, axis=1, b=alpha)
    x["logW"] = -x["logQ"].copy()
    return x


@pytest.fixture
def log_q(n_samples, n_it):
    return np.random.randn(n_samples, n_it)


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
    d = {k: np.random.randn(n_it).tolist() for k in keys}
    d["stopping_criteria"] = dict(
        log_dZ=np.arange(n_it),
        ratio=np.arange(n_it),
        ess=np.arange(n_it),
        kl=np.arange(n_it),
    )
    return d
