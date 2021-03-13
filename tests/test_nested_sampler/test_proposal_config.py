# -*- coding: utf-8 -*-
"""
Test the functions related to configuring proposal methods
"""
import numpy as np
import pytest

from nessai.proposal import RejectionProposal, AnalyticProposal, FlowProposal
from nessai.gw.proposal import GWFlowProposal, LegacyGWFlowProposal


@pytest.mark.parametrize('maximum, result',
                         [[None, np.inf], [False, 0], [100, 100]])
def test_uninformed_maximum(sampler, maximum, result):
    """
    Test to check that the proposal is correctly configured depending on
    the maximum number of uninformed iterations.
    """
    sampler.configure_uninformed_proposal(None, False, maximum, None)
    assert sampler.maximum_uninformed == result

    if maximum is False:
        assert sampler.uninformed_sampling is False
    else:
        assert sampler.uninformed_sampling is True


def test_uninformed_threshold(sampler):
    """Test the check uninformed threshold is set correctly"""
    sampler.configure_uninformed_proposal(None, False, None, 0.5)
    assert sampler.uninformed_acceptance_threshold == 0.5


def test_uninformed_threshold_default_below(sampler):
    """
    Test to check that the threshold is set to 10 times the acceptance
    if it is below 0.1.
    """
    sampler.acceptance_threshold = 0.05
    sampler.configure_uninformed_proposal(None, False, None, None)
    assert sampler.uninformed_acceptance_threshold == 0.5


@pytest.mark.parametrize('threshold', [0.1, 0.2])
def test_uninformed_threshold_default_(sampler, threshold):
    """
    Test to check that the threshold is set to the same value if it is above
    or equal to 0.1
    """
    sampler.acceptance_threshold = threshold
    sampler.configure_uninformed_proposal(None, False, None, None)
    assert sampler.uninformed_acceptance_threshold == threshold


def test_uninformed_no_analytic_priors(sampler):
    """
    Test to check that the correct proposal method is used without analytic
    priors.
    """
    sampler.configure_uninformed_proposal(None, False, None, None)
    assert isinstance(sampler._uninformed_proposal, RejectionProposal)


def test_uninformed_analytic_priors(sampler):
    """
    Test to check that the correct proposal method is used with analytic
    priors.
    """
    sampler.configure_uninformed_proposal(None, True, None, None)
    assert isinstance(sampler._uninformed_proposal, AnalyticProposal)


def test_uninformed_proposal_class(sampler):
    """Test using a custom proposal class"""
    from nessai.proposal.base import Proposal

    class TestProposal(Proposal):
        pass

    sampler.configure_uninformed_proposal(TestProposal, False, None, None)
    assert isinstance(sampler._uninformed_proposal, TestProposal)


def test_no_flow_proposal_class(sampler):
    """Test the default flow class"""
    sampler.configure_flow_proposal(None, {}, False)
    assert isinstance(sampler._flow_proposal, FlowProposal)


@pytest.mark.parametrize('flow_class, result_class',
                         [['FlowProposal', FlowProposal],
                          ['GWFlowProposal', GWFlowProposal],
                          ['LegacyGWFlowProposal', LegacyGWFlowProposal]])
def test_flow__class(flow_class, result_class, sampler):
    """Test the correct class is imported and used"""
    sampler.configure_flow_proposal(flow_class, {}, False)
    assert isinstance(sampler._flow_proposal, result_class)


def test_unknown_flow_class(sampler):
    """Test the to check the error raised if an uknown class is used"""
    with pytest.raises(RuntimeError) as excinfo:
        sampler.configure_flow_proposal('GWProposal', {}, False)
    assert 'Unknown' in str(excinfo.value)
