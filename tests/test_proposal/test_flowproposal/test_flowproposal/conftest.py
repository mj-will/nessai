# -*- coding: utf-8 -*-
from unittest.mock import create_autospec

import numpy as np
import pytest

from nessai.proposal.flowproposal import FlowProposal
from nessai.proposal.flowproposal.truncation import TruncationScheme


@pytest.fixture(params=[True, False])
def map_to_unit_hypercube(request):
    return request.param


@pytest.fixture()
def proposal(rng):
    proposal = create_autospec(FlowProposal, instance=True)
    proposal._initialised = False
    proposal.initialised = False
    proposal.accumulate_weights = False
    proposal.truncate_log_q = False
    proposal.enforce_likelihood_threshold = False
    proposal.map_to_unit_hypercube = False
    proposal.truncation_method = None
    proposal.truncation_methods = []
    proposal.truncation_kwargs = {}
    proposal._truncation_scheme = TruncationScheme()
    proposal.poolsize = 2000
    proposal.drawsize = 2000
    proposal.latent_temperature = None
    proposal.rng = rng
    return proposal


@pytest.fixture
def point():
    def _point(x, y, logl=0.0):
        out = np.zeros(1, dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")])
        out["x"] = x
        out["y"] = y
        out["logL"] = logl
        return out[0]

    return _point


@pytest.fixture
def samples():
    def _samples(values):
        out = np.zeros(
            len(values), dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")]
        )
        out["x"] = [v[0] for v in values]
        out["y"] = [v[1] for v in values]
        return out

    return _samples
