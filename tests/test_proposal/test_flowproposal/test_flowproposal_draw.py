# -*- coding: utf-8 -*-
"""Tests related to drawing new points from the pool."""
import numpy as np
import pytest
from unittest.mock import MagicMock, Mock

from nessai.livepoint import dict_to_live_points
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
    import datetime

    proposal.populated = False
    proposal.poolsize = 100
    proposal.population_time = datetime.timedelta()
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
    assert proposal.population_time.total_seconds() > 0.0

    proposal.populate.assert_called_once_with(1.0, N=100)

    assert proposal.update_poolsize_scale.called == update


def test_test_draw(proposal):
    """
    Test the method that tests the draw and populate methods when running.
    """
    test_point = dict_to_live_points(
        {"x": 1, "y": 2, "logP": -0.5}, non_sampling_parameters=False
    )
    new_point = dict_to_live_points(
        {"x": 3, "y": 4, "logP": -0.5}, non_sampling_parameters=False
    )
    proposal.model = Mock()
    proposal.model.new_point = MagicMock(return_value=test_point)
    proposal.model.log_prior = MagicMock(return_value=-0.5)
    proposal.populate = MagicMock()
    proposal.draw = MagicMock(return_value=new_point)
    proposal.reset = MagicMock()

    FlowProposal.test_draw(proposal)

    proposal.populate.assert_called_once_with(
        test_point, N=1, plot=False, r=1.0
    )
    proposal.reset.assert_called_once()
    proposal.draw.assert_called_once_with(test_point)


@pytest.mark.timeout(10)
@pytest.mark.flaky(reruns=3)
@pytest.mark.integration_test
def test_test_draw_integration(model, tmpdir):
    """Integration test for the test draw method"""
    proposal = FlowProposal(
        model,
        poolsize=10,
        output=tmpdir.mkdir("test_draw"),
        volume_fraction=0.5,
    )
    proposal.initialise()
    proposal.test_draw()
