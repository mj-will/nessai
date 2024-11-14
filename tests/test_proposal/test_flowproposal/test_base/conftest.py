# -*- coding: utf-8 -*-
from unittest.mock import create_autospec

import pytest

from nessai.proposal.flowproposal.base import BaseFlowProposal


@pytest.fixture(params=[True, False])
def map_to_unit_hypercube(request):
    return request.param


@pytest.fixture()
def proposal(rng):
    proposal = create_autospec(BaseFlowProposal, rng=rng)
    proposal._initialised = False
    proposal.map_to_unit_hypercube = False
    return proposal
