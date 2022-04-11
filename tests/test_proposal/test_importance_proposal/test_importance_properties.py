# -*- coding: utf-8 -*-
"""
Test the properties in ImportanceFlowProposal
"""
import numpy as np
import pytest

from nessai.proposal.importance import ImportanceFlowProposal


def test_total_samples_requested(proposal):
    proposal.n_requested = {'0': 50, '1': 100, '2': 200}
    out = ImportanceFlowProposal.total_samples_requested.__get__(proposal)
    assert out == 350


def test_total_samples_drawn(proposal):
    proposal.n_draws = {'0': 50, '1': 100, '2': 200}
    out = ImportanceFlowProposal.total_samples_drawn.__get__(proposal)
    assert out == 350


def test_n_proposals(proposal):
    proposal.n_draws = {'0': 100, '1': 100, '2': 100}
    out = ImportanceFlowProposal.n_proposals.__get__(proposal)
    assert out == 3


@pytest.mark.parametrize(
    "reweight, expected",
    [(True, 200), (False, 300)],
)
def test_normalisation_constant(proposal, reweight, expected):
    proposal.reweight_draws = reweight
    proposal.total_samples_requested = 200
    proposal.total_samples_drawn = 300
    out = ImportanceFlowProposal.normalisation_constant.__get__(proposal)
    assert out == expected


@pytest.mark.parametrize(
    "reweight, expected",
    [(True, {'0': 25, '1': 50}), (False, {'0': 50, '1': 100})],
)
def test_unnormalised_weights(proposal, reweight, expected):
    proposal.reweight_draws = reweight
    proposal.n_requested = {'0': 25, '1': 50}
    proposal.n_draws = {'0': 50, '1': 100}
    out = ImportanceFlowProposal.unnormalised_weights.__get__(proposal)
    assert out == expected


def test_poolsize(proposal):
    proposal.unnormalised_weights = {'intial': 50, '1': 100, '2': 200}
    out = ImportanceFlowProposal.poolsize.__get__(proposal)
    np.testing.assert_array_equal(out, np.array([50, 100, 200]))


def test_flow_config(proposal):
    cfg = {'max_epochs': 10}
    proposal._flow_config = cfg
    out = ImportanceFlowProposal.flow_config.__get__(proposal)
    assert out == cfg


@pytest.mark.parametrize(
    "reset_flows, expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (2, True),
        (4, False),
    ],
)
def test_reset_flow(proposal, reset_flows, expected):
    proposal.level_count = 2
    proposal.reset_flows = int(reset_flows)
    out = ImportanceFlowProposal._reset_flow.__get__(proposal)
    assert out == expected
