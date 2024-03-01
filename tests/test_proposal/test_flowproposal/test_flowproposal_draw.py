# -*- coding: utf-8 -*-
"""Tests related to drawing new points from the pool."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from nessai.proposal import FlowProposal


def test_draw_populated(proposal):
    """Test the draw method if the proposal is already populated"""
    proposal.populated = True
    proposal.samples = np.arange(3)
    proposal.indices = list(range(3))
    out = FlowProposal.draw(proposal, None)
    assert out == proposal.samples[2]
    assert proposal.indices == [0, 1]


def test_draw_populated_last_sample(proposal):
    """Test the draw method if the proposal is already populated but there
    is only one sample left.
    """
    proposal.populated = True
    proposal.samples = np.arange(3)
    proposal.indices = [0]
    out = FlowProposal.draw(proposal, None)
    assert out == proposal.samples[0]
    assert proposal.indices == []
    assert proposal.populated is False


@pytest.mark.parametrize("update", [False, True])
def test_draw_not_populated(proposal, update, wait):
    """Test the draw method when the proposal is not populated"""
    proposal.populated = False
    proposal.poolsize = 100
    proposal.samples = None
    proposal.indices = []
    proposal.update_poolsize = update
    proposal.update_poolsize_scale = MagicMock()
    proposal.ns_acceptance = 0.5

    def mock_populate(*args, **kwargs):
        wait()
        proposal.populated = True
        proposal.samples = np.arange(3)
        proposal.indices = list(range(3))

    proposal.populate = MagicMock(side_effect=mock_populate)

    out = FlowProposal.draw(proposal, 1.0)

    assert out == 2
    assert proposal.populated is True

    proposal.populate.assert_called_once_with(1.0, N=100)

    if update:
        proposal.update_poolsize_scale.assert_called_once()
    else:
        proposal.update_poolsize_scale.assert_not_called()
