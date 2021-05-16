# -*- coding: utf-8 -*-
"""
Test the functions related to configuring proposal methods
"""
import numpy as np
import pytest

from nessai.nestedsampler import NestedSampler
from nessai.proposal import (
    AnalyticProposal,
    AugmentedFlowProposal,
    RejectionProposal,
    FlowProposal
)
from nessai.gw.proposal import (
    AugmentedGWFlowProposal,
    GWFlowProposal,
    LegacyGWFlowProposal
)


@pytest.fixture
def sampler(sampler, tmpdir):
    sampler.output = tmpdir.mkdir('test')
    sampler.plot = False
    sampler.n_pool = None
    sampler.acceptance_threshold = 0.01
    return sampler


@pytest.mark.parametrize('maximum, result',
                         [[None, np.inf], [False, 0], [100, 100]])
def test_uninformed_maximum(sampler, maximum, result):
    """
    Test to check that the proposal is correctly configured depending on
    the maximum number of uninformed iterations.
    """
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, maximum, None)
    assert sampler.maximum_uninformed == result

    if maximum is False:
        assert sampler.uninformed_sampling is False
    else:
        assert sampler.uninformed_sampling is True


def test_uninformed_threshold(sampler):
    """Test the check uninformed threshold is set correctly"""
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, 0.5)
    assert sampler.uninformed_acceptance_threshold == 0.5


def test_uninformed_threshold_default_below(sampler):
    """
    Test to check that the threshold is set to 10 times the acceptance
    if it is below 0.1.
    """
    sampler.acceptance_threshold = 0.05
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, None)
    assert sampler.uninformed_acceptance_threshold == 0.5


@pytest.mark.parametrize('threshold', [0.1, 0.2])
def test_uninformed_threshold_default_(sampler, threshold):
    """
    Test to check that the threshold is set to the same value if it is above
    or equal to 0.1
    """
    sampler.acceptance_threshold = threshold
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, None)
    assert sampler.uninformed_acceptance_threshold == threshold


def test_uninformed_no_analytic_priors(sampler):
    """
    Test to check that the correct proposal method is used without analytic
    priors.
    """
    NestedSampler.configure_uninformed_proposal(
        sampler, None, False, None, None)
    assert isinstance(sampler._uninformed_proposal, RejectionProposal)


def test_uninformed_analytic_priors(sampler):
    """
    Test to check that the correct proposal method is used with analytic
    priors.
    """
    NestedSampler.configure_uninformed_proposal(
        sampler, None, True, None, None)
    assert isinstance(sampler._uninformed_proposal, AnalyticProposal)


def test_uninformed_proposal_class(sampler):
    """Test using a custom proposal class"""
    from nessai.proposal.base import Proposal

    class TestProposal(Proposal):
        pass

    NestedSampler.configure_uninformed_proposal(
        sampler, TestProposal, False, None, None)
    assert isinstance(sampler._uninformed_proposal, TestProposal)


def test_no_flow_proposal_class(sampler):
    """Test the default flow class"""
    NestedSampler.configure_flow_proposal(sampler, None, {}, False)
    assert isinstance(sampler._flow_proposal, FlowProposal)


@pytest.mark.parametrize('flow_class, result_class',
                         [['FlowProposal', FlowProposal],
                          ['AugmentedFlowProposal', AugmentedFlowProposal],
                          ['GWFlowProposal', GWFlowProposal],
                          ['AugmentedGWFlowProposal', AugmentedGWFlowProposal],
                          ['LegacyGWFlowProposal', LegacyGWFlowProposal]])
def test_flow__class(flow_class, result_class, sampler):
    """Test the correct class is imported and used"""
    NestedSampler.configure_flow_proposal(sampler, flow_class, {}, False)
    assert isinstance(sampler._flow_proposal, result_class)


def test_unknown_flow_class(sampler):
    """Test the to check the error raised if an uknown class is used"""
    with pytest.raises(RuntimeError) as excinfo:
        NestedSampler.configure_flow_proposal(sampler, 'GWProposal', {}, False)
    assert 'Unknown' in str(excinfo.value)
