# -*- coding: utf-8 -*-
import pytest
from unittest.mock import create_autospec

from nessai.proposal.flowproposal import FlowProposal


@pytest.fixture(params=[True, False])
def map_to_unit_hypercube(request):
    return request.param


@pytest.fixture()
def proposal():
    proposal = create_autospec(FlowProposal)
    proposal._initialised = False
    proposal.accumulate_weights = False
    proposal.map_to_unit_hypercube = False
    return proposal
