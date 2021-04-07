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


@pytest.mark.parametrize('generate', ['zeroes', 'gaussian'])
def test_rescaling(proposal, x, generate):
    """Test the rescaling method"""
    proposal._base_rescale = MagicMock(return_value=[x, np.zeros(x.size)])
    proposal.augment_names = ['e_1', 'e_2']
    proposal.augment_features = 2

    AugmentedFlowProposal._augmented_rescale(
        proposal, x, generate_augment=generate, test=True)

    proposal._base_rescale.assert_called_once_with(
        x, compute_radius=False, test=True)


@patch('numpy.random.randn', return_value=np.arange(10))
def test_rescaling_generate_none(mock, proposal, x):
    """Test the rescaling method with generate_augment=None"""
    proposal._base_rescale = MagicMock(return_value=[x, np.zeros(x.size)])
    proposal.augment_names = ['e_1', 'e_2']
    proposal.augment_features = 2
    proposal.generate_augment = 'gaussian'

    AugmentedFlowProposal._augmented_rescale(
        proposal, x, generate_augment=None, test=True, compute_radius=False)

    proposal._base_rescale.assert_called_once_with(
        x, compute_radius=False, test=True)
    assert mock.call_count == 2


@patch('scipy.stats.norm.logpdf')
def test_augmented_prior(mock, proposal, x):
    """Test the augmented prior"""
    proposal.augment_names = ['e_1', 'e_2']
    AugmentedFlowProposal.augmented_prior(proposal, x)
    np.testing.assert_array_equal(x['e_1'], mock.call_args_list[0][0][0])
    np.testing.assert_array_equal(x['e_2'], mock.call_args_list[1][0][0])
