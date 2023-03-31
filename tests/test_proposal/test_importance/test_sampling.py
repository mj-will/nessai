"""Tests related to sampling from ImportanceFlowProposal"""
from unittest.mock import MagicMock

from nessai.proposal.importance import ImportanceFlowProposal as IFP
import numpy as np
import pytest


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.usefixtures("ins_parameters")
def test_draw_from_prior(ifp, x, n):
    """Test drawing from the prior"""
    dims = 2
    n_proposals = 4
    x_prime = np.random.randn(n, 2)
    log_j = np.random.rand(n)
    log_Q = np.random.randn(n)
    log_q = np.random.randn(n_proposals, n)
    ifp.model.dims = dims
    ifp.to_prime = MagicMock(return_value=(x_prime, log_j))
    ifp.inverse_rescale = MagicMock(return_value=(x, None))
    ifp.compute_log_Q = MagicMock(return_value=(log_Q, log_q))

    x_out, log_q_out = IFP.draw_from_prior(ifp, n)
    assert log_q_out.shape == (n_proposals, n)

    assert x_out is x
