"""Tests methods for computing log_prob etc"""
from unittest.mock import MagicMock

from nessai.livepoint import live_points_to_array
from nessai.proposal.importance import ImportanceFlowProposal as IFP
import numpy as np
import pytest


@pytest.mark.parametrize("p_it, q_it", [(None, None), (-1, 0), (3, 4)])
def test_kl_between_proposals(ifp, model, p_it, q_it, x):
    def rescale(x):
        x = model.to_unit_hypercube(x)
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def get_proposal_log_prob(it):
        def log_prob(x):
            if it == -1:
                return np.zeros(len(x))
            else:
                return np.log(np.random.rand(len(x)))

        return log_prob

    ifp.flow = MagicMock()
    ifp.flow.n_models = 15
    ifp.rescale = MagicMock(side_effect=rescale)
    ifp.get_proposal_log_prob = MagicMock(side_effect=get_proposal_log_prob)

    out = IFP.compute_kl_between_proposals(ifp, x, p_it, q_it)

    assert isinstance(out, float)
    assert np.isfinite(out)
