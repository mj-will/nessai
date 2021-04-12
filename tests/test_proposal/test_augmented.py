# -*- coding: utf-8 -*-
"""Test the augment proposal"""
import numpy as np
import pytest
from unittest.mock import create_autospec, MagicMock, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import AugmentedFlowProposal


@pytest.fixture
def proposal():
    return create_autospec(AugmentedFlowProposal)


@pytest.fixture
def x():
    _x = np.concatenate([np.random.rand(10, 2), np.random.randn(10, 2)],
                        axis=1)
    return numpy_array_to_live_points(_x, ['x', 'y', 'e_1', 'e_2'])


def test_init(model):
    """Test the init function"""
    AugmentedFlowProposal(model, poolsize=100)


@patch('nessai.proposal.augmented.FlowModel')
@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
def test_initialise(mock_osexists, mock_makedirs, mock_flow, proposal):
    """Test the method the initialise the proposal"""
    proposal.output = 'output'
    proposal.rescaled_dims = 4
    proposal.augment_dims = 2
    proposal.expansion_fraction = 1.0
    proposal.flow_config = {'model_config': {}}
    proposal.set_rescaling = MagicMock()
    proposal.verify_rescaling = MagicMock()
    AugmentedFlowProposal.initialise(proposal)

    assert proposal.initialised is True
    assert proposal.fuzz == (2 ** 0.25)
    assert proposal.flow_config['model_config']['kwargs']['mask'].tolist() == \
        [1, 1, -1, -1]
    proposal.set_rescaling.assert_called_once()
    proposal.verify_rescaling.assert_called_once()
    mock_flow.assert_called_once()
    proposal.flow.initialise.assert_called_once()
    mock_osexists.assert_called_once_with('output')
    mock_makedirs.assert_called_once_with('output', exist_ok=True)


@patch('nessai.proposal.FlowProposal.set_rescaling')
def test_set_rescaling(mock, proposal):
    """Test the set rescaling method"""
    proposal.names = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime']
    proposal.rescale_parameters = proposal.names
    proposal.augment_dims = 2
    AugmentedFlowProposal.set_rescaling(proposal)

    assert proposal.names == ['x', 'y', 'e_0', 'e_1']
    assert proposal.rescaled_names == ['x_prime', 'y_prime', 'e_0', 'e_1']
    assert proposal.augment_names == ['e_0', 'e_1']
    mock.assert_called_once()


@pytest.mark.parametrize('generate', ['zeroes', 'gaussian'])
@patch('numpy.random.randn', return_value=np.ones(10))
@patch('numpy.zeros', return_value=np.ones(10))
def test_rescaling(mock_zeros, mock_randn, proposal, x, generate):
    """Test the rescaling method"""
    proposal._base_rescale = MagicMock(return_value=[x, np.ones(x.size)])
    proposal.augment_names = ['e_1']
    proposal.augment_dims = 1

    AugmentedFlowProposal._augmented_rescale(
        proposal, x, generate_augment=generate, test=True)

    proposal._base_rescale.assert_called_once_with(
        x, compute_radius=False, test=True)

    if generate == 'zeroes':
        mock_zeros.assert_called_once_with(x.size)
    else:
        mock_randn.assert_called_once_with(x.size)


@pytest.mark.parametrize('compute_radius', [False, True])
@patch('numpy.random.randn', return_value=np.arange(10))
@patch('numpy.zeros', return_value=np.ones(10))
def test_rescaling_generate_none(mock_zeros, mock_randn, proposal, x,
                                 compute_radius):
    """Test the rescaling method with generate_augment=None"""
    proposal._base_rescale = MagicMock(return_value=[x, np.ones(x.size)])
    proposal.augment_names = ['e_1']
    proposal.augment_dims = 1
    proposal.generate_augment = 'zeros'

    AugmentedFlowProposal._augmented_rescale(
        proposal, x, generate_augment=None, test=True,
        compute_radius=compute_radius)

    proposal._base_rescale.assert_called_once_with(
        x, compute_radius=compute_radius, test=True)

    if not compute_radius:
        mock_randn.assert_called_once_with(x.size)
    else:
        mock_zeros.assert_called_once_with(x.size)


def test_rescaling_generate_unknown(proposal, x):
    """Test the rescaling method with with an unknown method for generate"""
    proposal._base_rescale = MagicMock(return_value=[x, np.ones(x.size)])

    with pytest.raises(RuntimeError) as excinfo:
        AugmentedFlowProposal._augmented_rescale(
            proposal, x, generate_augment='ones')
    assert 'Unknown method' in str(excinfo.value)


@patch('scipy.stats.norm.logpdf')
def test_augmented_prior(mock, proposal, x):
    """Test the augmented prior"""
    proposal.marginalise_augment = False
    proposal.augment_names = ['e_1', 'e_2']
    AugmentedFlowProposal.augmented_prior(proposal, x)
    np.testing.assert_array_equal(x['e_1'], mock.call_args_list[0][0][0])
    np.testing.assert_array_equal(x['e_2'], mock.call_args_list[1][0][0])


@pytest.mark.parametrize('marg', [False, True])
def test_log_prior(proposal, x, marg):
    """Test the augmented prior"""
    proposal.marginalise_augment = marg
    proposal.augmented_prior = MagicMock()
    proposal.model = MagicMock()
    proposal.model.names = ['x', 'y']
    proposal.model.log_prior = MagicMock()

    AugmentedFlowProposal.log_prior(proposal, x)

    proposal.model.log_prior.assert_called_once()
    print(proposal.augmented_prior.called)
    assert proposal.augmented_prior.called is not marg


def test_marginalise_augment(proposal):
    """Test the marginalise augment function"""
    proposal.n_marg = 5
    proposal.augment_dims = 2
    x = np.concatenate([np.random.randn(3, 2), np.zeros((3, 2))], axis=1)
    z = np.random.randn(15, 4)
    log_prob = np.random.randn(15)
    proposal.flow = MagicMock()
    proposal.flow.forward_and_log_prob = \
        MagicMock(return_value=[z, log_prob])
    log_prob_out = AugmentedFlowProposal._marginalise_augment(proposal, x)

    assert len(log_prob_out) == 3


@pytest.mark.parametrize('log_p', [np.ones(2), np.array([-1, np.inf])])
@pytest.mark.parametrize('marg', [False, True])
def test_backward_pass(proposal, model, log_p, marg):
    """Test the backward pass method"""
    n = 2
    acc = int(np.isfinite(log_p).sum())
    x = np.random.randn(n, model.dims)
    z = np.random.randn(n, model.dims)
    proposal._marginalise_augment = MagicMock(return_value=log_p)
    proposal.inverse_rescale = \
        MagicMock(side_effect=lambda a: (a, np.ones(a.size)))
    proposal.rescaled_names = model.names
    proposal.alt_dist = None
    proposal.check_prior_bounds = MagicMock(side_effect=lambda a, b: (a, b))
    proposal.flow = MagicMock()
    proposal.flow.sample_and_log_prob = \
        MagicMock(return_value=[x, log_p])

    proposal.marginalise_augment = marg

    x_out, log_p = AugmentedFlowProposal.backward_pass(proposal, z)

    assert len(x_out) == acc
    proposal.inverse_rescale.assert_called_once()
    proposal.flow.sample_and_log_prob.assert_called_once_with(
        z=z, alt_dist=None)

    assert proposal._marginalise_augment.called is marg
