# -*- coding: utf-8 -*-
import pytest
from unittest.mock import create_autospec

from nessai.proposal import FlowProposal


@pytest.fixture()
def proposal():
    proposal = create_autospec(FlowProposal)
    proposal._initialised = False
    proposal.accumulate_weights = False
    return proposal
