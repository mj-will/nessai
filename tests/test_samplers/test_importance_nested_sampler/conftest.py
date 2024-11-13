from unittest.mock import create_autospec

import numpy as np
import pytest
from scipy.special import logsumexp

from nessai.evidence import _INSIntegralState
from nessai.model import Model
from nessai.proposal.importance import ImportanceFlowProposal
from nessai.samplers.importancesampler import (
    ImportanceNestedSampler,
    OrderedSamples,
)


@pytest.fixture(scope="module", params=[False, True])
def iid(request):
    return request.param


@pytest.fixture
def ins(rng):
    obj = create_autospec(ImportanceNestedSampler)
    obj.model = create_autospec(Model)
    obj.training_samples = create_autospec(OrderedSamples)
    obj.training_samples.state = create_autospec(_INSIntegralState)
    obj.iid_samples = None
    obj.rng = rng
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
def samples(model, n_samples, n_it, log_q, ins_parameters):
    x = model.sample_unit_hypercube(n_samples)
    x["it"] = np.random.randint(-1, n_it - 1, size=len(x))
    xx = model.from_unit_hypercube(x)
    x["logL"] = model.log_likelihood(xx)
    x["logP"] = model.log_prior(xx)

    alpha = np.bincount(
        x["it"] + np.abs(x["it"].min()), minlength=n_it
    ).astype(float)
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
        "min_log_likelihood",
        "max_log_likelihood",
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
