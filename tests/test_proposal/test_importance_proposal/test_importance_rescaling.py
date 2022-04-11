# -*- coding: utf-8 -*-
"""
Test rescaling for the importance proposal.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from nessai.proposal.importance import ImportanceFlowProposal


@pytest.mark.parametrize('reparam', ['logit', 'gaussian_cdf', None])
def test_to_prime_single_point(proposal, reparam):
    """Assert `to_prime` works with a single 1D point."""
    x = np.random.rand(10)
    proposal.reparam = reparam
    x_prime, log_j = ImportanceFlowProposal.to_prime(proposal, x)
    assert len(x_prime) == 1
    assert len(log_j) == 1


@pytest.mark.parametrize('reparam', ['logit', 'gaussian_cdf', None])
def test_from_prime_single_point(proposal, reparam):
    """Assert `from_prime` works with a single 1D point."""
    x = np.random.rand(10)
    proposal.reparam = reparam
    x_prime, log_j = ImportanceFlowProposal.from_prime(proposal, x)
    assert len(x_prime) == 1
    assert len(log_j) == 1


def test_rescale(proposal):
    """Test the rescale method"""

    x = np.array([1, 2])
    x_unit = np.array([3, 4])
    x_array = np.array([5, 6])
    x_prime = np.array([7, 8])
    log_j = np.array([9, 10])

    proposal.model.to_unit_hypercube = MagicMock(return_value=x_unit)
    proposal.model.names = ['x']

    proposal.to_prime = MagicMock(return_value=(x_prime, log_j))

    with patch(
        'nessai.proposal.importance.live_points_to_array',
        return_value=x_array,
    ) as mock_conv:
        out = ImportanceFlowProposal.rescale(proposal, x)

    proposal.model.to_unit_hypercube.assert_called_once_with(x)
    mock_conv.assert_called_once_with(x_unit, ['x'])
    assert out == (x_prime, log_j)


def test_inverse_rescale(proposal):
    """Test the inverse rescale method"""

    x = np.array([1, 2])
    x_unit = np.array([3, 4])
    x_array = np.array([5, 6])
    x_prime = np.array([7, 8])
    log_j = np.array([9, 10])

    proposal.from_prime = MagicMock(return_value=(x_array, log_j))
    proposal.clip = False

    proposal.model.from_unit_hypercube = MagicMock(return_value=x)
    proposal.model.names = ['x']

    with patch(
        'nessai.proposal.importance.numpy_array_to_live_points',
        return_value=x_unit,
    ) as mock_conv:
        out = ImportanceFlowProposal.inverse_rescale(proposal, x_prime)

    proposal.model.from_unit_hypercube.assert_called_once_with(x_unit)
    mock_conv.assert_called_once_with(x_array, ['x'])
    assert out == (x, log_j)
