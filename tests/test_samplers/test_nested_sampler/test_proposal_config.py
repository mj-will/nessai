# -*- coding: utf-8 -*-
"""
Test the functions related to configuring proposal methods
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from nessai.samplers.nestedsampler import NestedSampler
from nessai.proposal import (
    AnalyticProposal,
    AugmentedFlowProposal,
    RejectionProposal,
    FlowProposal,
)
from nessai.gw.proposal import (
    AugmentedGWFlowProposal,
    GWFlowProposal,
)


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


def test_no_flow_proposal_class(sampler):
    """Test the default flow class"""
    NestedSampler.configure_flow_proposal(sampler, None, {}, False)
    assert isinstance(sampler._flow_proposal, FlowProposal)


@pytest.mark.parametrize(
    "flow_class, result_class",
    [
        ["FlowProposal", FlowProposal],
        ["AugmentedFlowProposal", AugmentedFlowProposal],
        ["GWFlowProposal", GWFlowProposal],
        ["AugmentedGWFlowProposal", AugmentedGWFlowProposal],
        ["flowproposal", FlowProposal],
        ["augmentedflowproposal", AugmentedFlowProposal],
        ["gwflowproposal", GWFlowProposal],
        ["augmentedgwflowproposal", AugmentedGWFlowProposal],
    ],
)
def test_flow__class(flow_class, result_class, sampler):
    """Test the correct class is imported and used"""
    NestedSampler.configure_flow_proposal(sampler, flow_class, {}, False)
    assert isinstance(sampler._flow_proposal, result_class)


def test_unknown_flow_class(sampler):
    """Test to check the error raised if an unknown class is used"""
    with pytest.raises(ValueError) as excinfo:
        NestedSampler.configure_flow_proposal(sampler, "GWProposal", {}, False)
    assert "Unknown flow class" in str(excinfo.value)


def test_flow_class_not_subclass(sampler):
    """
    Test to check an error is raised in the class does not inherit from
    FlowProposal
    """

    class FakeProposal:
        pass

    with pytest.raises(RuntimeError) as excinfo:
        NestedSampler.configure_flow_proposal(sampler, FakeProposal, {}, False)
    assert "inherits" in str(excinfo.value)


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
