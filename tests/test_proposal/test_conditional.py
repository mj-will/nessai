# -*- coding: utf-8 -*-
"""Test the conditional proposal"""
from nessai.livepoint import numpy_array_to_live_points
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


def test_check_state(proposal):
    """Test the check state method.

    Checks that the parent method has been called and that the min and max
    likelihoods are set.
    """
    proposal.update_bounds = True
    x = {'logL': np.arange(10)}
    with patch('nessai.proposal.conditional.FlowProposal.check_state') as m:
        ConditionalFlowProposal.check_state(proposal, x)
    m.assert_called_once_with(x)
    assert proposal._min_logL == 0
    assert proposal._max_logL == 9


def test_configure_likelihood_parameter(proposal):
    """Make sure the likelihood parameter is correctly added"""
    proposal.conditional_likelihood = True
    proposal.conditional_parameters = ['c1']
    with patch('nessai.proposal.conditional.InterpolatedDistribution') as mock:
        ConditionalFlowProposal.configure_likelihood_parameter(proposal)
    mock.assert_called_once_with('logL', rescale=False)
    assert proposal.likelihood_index == 1
    assert proposal.conditional_parameters == ['c1', 'logL']


def test_update_flow_config(proposal):
    """Test updating the flow config.

    If the proposal is conditional, this should add `context_features`.
    """
    proposal.conditional = True
    proposal.flow_config = {'model_config': {'kwargs': {}}}
    with patch('nessai.proposal.conditional.FlowProposal.update_flow_config') \
            as m:
        ConditionalFlowProposal.update_flow_config(proposal)

    m.assert_called_once()


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


def test_train_on_data(proposal):
    """Test the train on data method"""
    output = 'out'
    proposal.rescaled_names = ['a', 'b']
    a = np.random.randn(10, 2)
    x_prime = numpy_array_to_live_points(a, proposal.rescaled_names)
    context = np.arange(10)
    proposal.get_context = MagicMock(return_value=context)
    proposal.train_context = MagicMock()
    proposal._plot_training = False
    proposal.flow = MagicMock()
    proposal.flow.train = MagicMock()
    with patch('nessai.proposal.conditional.live_points_to_array',
               return_value=a):
        ConditionalFlowProposal.train_on_data(proposal, x_prime, output)

    proposal.get_context.assert_called_once_with(x_prime)
    proposal.train_context.assert_called_once_with(context)
    proposal.flow.train.assert_called_once_with(
        a, context=context, output=output, plot=False)


@pytest.mark.parametrize('update', [False, True])
def test_train_context(proposal, update):
    """Test training on the context"""
    proposal.conditional_likelihood = True
    context = np.array([[1, 2], [3, 4]])
    proposal.update_bounds = update
    proposal.likelihood_index = 1
    proposal.likelihood_distribution = MagicMock()
    proposal.likelihood_distribution.update_samples = MagicMock()

    ConditionalFlowProposal.train_context(proposal, context)

    print(proposal.likelihood_distribution.update_samples.call_args[0])
    np.testing.assert_array_equal(
        proposal.likelihood_distribution.update_samples.call_args[0][0],
        np.array([2, 4])
    )
    assert (proposal.likelihood_distribution.update_samples.
            call_args[1]['reset']) is update


def test_sample_context_parameters(proposal):
    """Test sampling context parameters"""
    c = np.arange(10)
    proposal.conditional_likelihood = True
    proposal.conditional_dims = 2
    proposal.likelihood_index = 1
    proposal.likelihood_distribution = MagicMock()
    proposal.likelihood_distribution.sample = MagicMock(return_value=c)
    proposal.rescaled_worst_logL = -0.5

    out = ConditionalFlowProposal.sample_context_parameters(proposal, 10)

    proposal.likelihood_distribution.sample.assert_called_once_with(
        10, min_logL=-0.5
    )

    np.testing.assert_array_equal(out[:, 1], c)


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


def test_get_context_not_conditional(proposal):
    """Assert `get_context` returns None if conditional=False"""
    proposal.conditional = False
    assert ConditionalFlowProposal.get_context(proposal, 2) is None


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
