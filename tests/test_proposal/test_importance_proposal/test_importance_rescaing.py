# -*- coding: utf-8 -*-
"""
Test rescaling for the importance proposal.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.proposal.importance import ImportanceFlowProposal


@pytest.fixture
def proposal():
    return create_autospec(ImportanceFlowProposal)


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
