# -*- coding: utf-8 -*-
"""Test the conditional proposal"""
import numpy as np
import pytest
from unittest.mock import create_autospec, patch, MagicMock

from nessai.proposal.conditional import ConditionalFlowProposal


@pytest.fixture
def proposal():
    return create_autospec(ConditionalFlowProposal)


@pytest.fixture
def proposal_init(model, tmpdir):
    return ConditionalFlowProposal(
        model,
        output=str(tmpdir.mkdir('conditional')),
        conditional_likelihood=True,
        poolsize=10,
        rescale_parameters=True
    )


@pytest.mark.parametrize('n', [0, 2])
def test_conditional_dims(proposal, n):
    """Test to make sure the correct dims are returned"""
    proposal.conditional_parameters = n * ['p']
    assert ConditionalFlowProposal.conditional_dims.__get__(proposal) == n


def test_rescaled_worst_logL(proposal):
    """Test the rescaling"""
    proposal.worst_logL = 0.5
    proposal._min_logL = -1.0
    proposal._max_logL = 1.0
    logL = ConditionalFlowProposal.rescaled_worst_logL.__get__(proposal)
    assert logL == 0.75


def test_rescaled_worst_logL_none(proposal):
    """Test the rescaled logL property when logL is not defined"""
    proposal.worst_logL = None
    logL = ConditionalFlowProposal.rescaled_worst_logL.__get__(proposal)
    assert logL is None


def test_set_rescaling(proposal):
    """Test the set rescaling method.

    Checks to make sure the parent method is called.
    """
    with patch('nessai.proposal.conditional.FlowProposal.set_rescaling') as m:
        ConditionalFlowProposal.set_rescaling(proposal)
    m.assert_called_once()


def test_configure_likelihood_parameter(proposal):
    """Make sure the likelihood parameter is correctly added"""
    proposal.conditional_likelihood = True
    proposal.conditional_parameters = ['c1']
    with patch('nessai.proposal.conditional.InterpolatedDistribution') as mock:
        ConditionalFlowProposal.configure_likelihood_parameter(proposal)
    mock.assert_called_once_with('logL', rescale=False)
    assert proposal.likelihood_index == 1
    assert proposal.conditional_parameters == ['c1', 'logL']


def test_reset_reparameterisations(proposal):
    """Test to make sure reparameterisation is reset."""
    proposal._min_logL = -1.0
    proposal._max_logL = 1.0
    with patch('nessai.proposal.conditional.FlowProposal.'
               'reset_reparameterisation') as mock:
        ConditionalFlowProposal.reset_reparameterisation(proposal)
    mock.assert_called_once()
    assert proposal._min_logL is None
    assert proposal._max_logL is None


def test_get_context_likelihood(proposal):
    """Assert `get_context` returns the correct likelihood context"""
    proposal.conditional = True
    proposal.conditional_dims = 1
    proposal.conditional_likelihood = True
    proposal.likelihood_index = 0
    proposal._min_logL = 0.0
    proposal._max_logL = 10.0
    x = np.array([1, 2], dtype=[('logL', 'f8')])
    c = ConditionalFlowProposal.get_context(proposal, x)

    assert np.array_equal(c, np.array([[0.1], [0.2]]))


def test_get_context_no_conditionals(proposal):
    """Assert `get_context` returns None with no conditional parameters but \
        conditional=True.

    This should not happen during use.
    """
    proposal.conditional = True
    proposal.conditional_likelihood = False
    proposal.conditional_dims = 0
    x = np.array([1, 2])
    c = ConditionalFlowProposal.get_context(proposal, x)
    assert c is None


def test_forward_pass(proposal):
    """Test the forward pass method"""
    proposal.conditional = True
    c = np.array([3, 4])
    proposal.get_context = MagicMock(return_value=c)
    x = np.array([1, 2])
    with patch('nessai.proposal.conditional.FlowProposal.forward_pass',
               return_value=(1, 2)) \
            as mock:
        out = ConditionalFlowProposal.forward_pass(
            proposal, x, context=None, compute_radius=True)

    mock.assert_called_once_with(x, context=c, compute_radius=True)
    proposal.get_context.assert_called_once_with(x)
    assert out == (1, 2)


def test_backward_pass(proposal):
    """Test the backward pass method"""
    proposal.conditional = True
    c = np.array([3, 4])
    proposal.sample_context_parameters = MagicMock(return_value=c)
    x = np.array([1, 2])
    with patch('nessai.proposal.conditional.FlowProposal.backward_pass',
               return_value=(1, 2)) \
            as mock:
        out = ConditionalFlowProposal.backward_pass(
            proposal, x, context=None, compute_radius=True)

    mock.assert_called_once_with(x, context=c, compute_radius=True)
    proposal.sample_context_parameters.assert_called_once_with(x.size)
    assert out == (1, 2)


@pytest.mark.integration_test
def test_reset_reparameterisations_integration(proposal_init):
    """Test to make reset works with parent class"""
    proposal._min_logL = -1.0
    proposal._max_logL = 1.0
    proposal_init.reset_reparameterisation()
    assert proposal_init._min_logL is None
    assert proposal_init._max_logL is None


@pytest.mark.integration_test
def test_conditional_init(proposal_init):
    """Integration test for the init method and initialise methods"""
    proposal_init.initialise()

    assert proposal_init.conditional is True
    assert proposal_init.rescaled_names == ['x_prime', 'y_prime']
    assert proposal_init.conditional_dims == 1
    assert proposal_init.conditional_parameters == ['logL']
    assert (proposal_init.
            flow_config['model_config']['kwargs']['context_features'] == 1)
