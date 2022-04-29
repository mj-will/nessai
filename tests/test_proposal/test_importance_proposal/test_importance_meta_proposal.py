# -*- coding: utf-8 -*-
"""
Tests for the meta-proposal in the importance proposal
"""
from unittest.mock import MagicMock

import numpy as np
from scipy.special import logsumexp

from nessai.flowmodel import CombinedFlowModel
from nessai.proposal import ImportanceFlowProposal as IFP


def test_compute_log_Q(proposal):
    """Test computing the log-meta proposal prob."""
    x_prime = np.random.randn(10, 4)
    log_q_flows = np.random.randn(10, 3)
    log_j = np.random.randn(10)
    log_q = np.concatenate(
        [np.zeros((10, 1)), log_q_flows + log_j[:, np.newaxis]], axis=1,
    )
    counts = np.array([4, 2, 2, 2])
    log_Q = logsumexp(log_q, b=counts, axis=1)

    proposal.n_proposals = 4
    proposal.poolsize = counts
    proposal.flow = MagicMock(spec=CombinedFlowModel)
    proposal.flow.log_prob_all = MagicMock(return_value=log_q_flows)
    proposal.flow.n_models = 3

    log_Q_out, log_q_out = IFP.compute_log_Q(
        proposal, x_prime, log_j=log_j,
    )

    proposal.flow.log_prob_all.assert_called_once_with(
        x_prime, exclude_last=False
    )

    np.testing.assert_array_equal(log_q_out, log_q)
    np.testing.assert_array_equal(log_Q_out, log_Q)
