# -*- coding: utf-8 -*-
"""
Test the functions related to configuring proposal methods
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai.proposal import (
    AnalyticProposal,
    RejectionProposal,
)
from nessai.samplers.nestedsampler import NestedSampler


@pytest.fixture
def sampler(sampler, tmpdir):
    sampler.output = tmpdir.mkdir("test")
    sampler.plot = False
    sampler.n_pool = None
    sampler.acceptance_threshold = 0.01
    return sampler


@pytest.mark.parametrize(
    "maximum, result", [[None, 200], [False, 0], [100, 100], ["inf", np.inf]]
)
def test_uninformed_maximum(sampler, maximum, result):
    """
    Test to check that the proposal is correctly configured depending on
    the maximum number of uninformed iterations.
    """
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, maximum, None
    )
    assert sampler.maximum_uninformed == result

    if maximum is False:
        assert sampler.uninformed_sampling is False
    else:
        assert sampler.uninformed_sampling is True


def test_uninformed_threshold(sampler):
    """Test the check uninformed threshold is set correctly"""
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, 0.5
    )
    assert sampler.uninformed_acceptance_threshold == 0.5


def test_uninformed_threshold_default_below(sampler):
    """
    Test to check that the threshold is set to 10 times the acceptance
    if it is below 0.1.
    """
    sampler.acceptance_threshold = 0.05
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, None
    )
    assert sampler.uninformed_acceptance_threshold == 0.5


@pytest.mark.parametrize("threshold", [0.1, 0.2])
def test_uninformed_threshold_default_(sampler, threshold):
    """
    Test to check that the threshold is set to the same value if it is above
    or equal to 0.1
    """
    sampler.acceptance_threshold = threshold
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, None
    )
    assert sampler.uninformed_acceptance_threshold == threshold


def test_uninformed_no_analytic_priors(sampler):
    """
    Test to check that the correct proposal method is used without analytic
    priors.
    """
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, None
    )
    assert isinstance(sampler._uninformed_proposal, RejectionProposal)


def test_uninformed_analytic_priors(sampler):
    """
    Test to check that the correct proposal method is used with analytic
    priors.
    """
    NestedSampler.configure_uninformed_proposal(
        sampler, None, True, None, None
    )
    assert isinstance(sampler._uninformed_proposal, AnalyticProposal)


def test_uninformed_proposal_class(sampler):
    """Test using a custom proposal class"""
    from nessai.proposal.base import Proposal

    class TestProposal(Proposal):
        def draw(self, point):
            pass

    NestedSampler.configure_uninformed_proposal(
        sampler, TestProposal, False, None, None
    )
    assert isinstance(sampler._uninformed_proposal, TestProposal)
    assert sampler._uninformed_proposal.rng is sampler.rng


def test_configure_flow_proposal(sampler, rng):
    fake_class = MagicMock()
    fake_class.__name__ = "fake_class"
    flow_config = dict(patience=10)
    kwargs = dict(test=True, invalid=True)
    expected_kwargs = dict(test=True, poolsize=sampler.nlive)

    with (
        patch(
            "nessai.samplers.nestedsampler.get_flow_proposal_class",
            return_value=fake_class,
        ) as mock_get,
        patch(
            "nessai.samplers.nestedsampler.check_proposal_kwargs",
            return_value=expected_kwargs,
        ) as mock_check,
    ):
        NestedSampler.configure_flow_proposal(
            sampler, None, flow_config, False, **kwargs
        )

    mock_get.assert_called_once_with(None)
    mock_check.assert_called_once_with(
        fake_class,
        {**kwargs, "poolsize": sampler.nlive},
    )

    fake_class.assert_called_once_with(
        sampler.model,
        flow_config=flow_config,
        output=os.path.join(sampler.output, "proposal", ""),
        plot=False,
        rng=rng,
        **expected_kwargs,
    )


@pytest.mark.parametrize("val", [(0.1, 10), (0.8, 200)])
def test_proposal_switch(sampler, val):
    """Test the method for switching proposals"""
    sampler.mean_block_acceptance = 0.5
    sampler.mean_acceptance = val[0]
    sampler.uninformed_acceptance_threshold = 0.5
    sampler.iteration = val[1]
    sampler.maximum_uninformed = 100
    sampler._flow_proposal = MagicMock()
    sampler._flow_proposal.n_pool = 2
    sampler._flow_proposal.configure_pool = MagicMock()
    sampler._uninformed_proposal = MagicMock()
    sampler._uninformed_proposal.pool = True
    sampler._uninformed_proposal.close_pool = MagicMock()
    sampler.proposal = sampler._uninformed_proposal

    assert NestedSampler.check_proposal_switch(sampler) is True

    assert sampler.uninformed_sampling is False
    assert sampler.proposal == sampler._flow_proposal


def test_proposal_no_switch(sampler):
    """Ensure proposal is not switched"""
    sampler.mean_acceptance = 0.5
    sampler.uninformed_acceptance_threshold = 0.1
    sampler.iteration = 10
    sampler.maximum_uninformed = 100
    assert NestedSampler.check_proposal_switch(sampler) is False


def test_proposal_already_switched(sampler):
    """Test switching when proposal is already switched"""
    sampler.mean_acceptance = 0.5
    sampler.uninformed_acceptance_threshold = 0.1
    sampler.mean_block_acceptance = 0.5
    sampler.iteration = 10
    sampler.maximum_uninformed = 100
    sampler._flow_proposal = MagicMock()
    sampler._flow_proposal.ns_acceptance = 0.2
    sampler.uninformed_sampling = False
    sampler.proposal = sampler._flow_proposal
    assert NestedSampler.check_proposal_switch(sampler, force=True) is True
    assert sampler.proposal is sampler._flow_proposal
    # If proposal was switched again, acceptance would change
    assert sampler.proposal.ns_acceptance == 0.2
