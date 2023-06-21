"""Tests methods for computing log_prob etc"""
from unittest.mock import MagicMock, create_autospec

from nessai.livepoint import live_points_to_array
from nessai.flowmodel.importance import ImportanceFlowModel
from nessai.proposal.importance import ImportanceFlowProposal as IFP
import numpy as np
from scipy.special import logsumexp
import pytest


@pytest.fixture()
def ifp(ifp):
    ifp.flow = create_autospec(ImportanceFlowModel)
    return ifp


def test_compute_log_Q(ifp, x_prime):
    n_flows = 3
    ifp.poolsize = np.array([5, 10, 15, 20])
    ifp.flow.n_models = n_flows
    ifp.n_proposals = n_flows + 1

    log_j = np.log(np.random.rand(len(x_prime)))

    def log_prob_all(x, exclude_last):
        assert exclude_last is False
        return np.log(np.random.rand(len(x), n_flows))

    ifp.flow.log_prob_all = MagicMock(side_effect=log_prob_all)

    log_Q, log_q = IFP.compute_log_Q(ifp, x_prime, log_j=log_j)

    assert len(log_Q) == len(x_prime)
    assert log_q.shape == (len(x_prime), n_flows + 1)
    assert all(log_q[:, 0] == 0)

    weights = ifp.poolsize / ifp.poolsize.sum()
    expected_log_Q = logsumexp(log_q, b=weights, axis=1)
    np.testing.assert_array_equal(log_Q, expected_log_Q)


def test_compute_log_Q_exclude_last(ifp, x_prime):
    n_flows = 3
    n = 30
    ifp.poolsize = np.array([5, 10, 15, 20])
    ifp.flow.n_models = n_flows
    ifp.n_proposals = n_flows + 1

    log_q_current = np.log(np.random.rand(len(x_prime)))
    log_j = np.log(np.random.rand(len(x_prime)))

    def log_prob_all(x, exclude_last):
        assert exclude_last is True
        return np.log(np.random.rand(len(x), n_flows - 1))

    ifp.flow.log_prob_all = MagicMock(side_effect=log_prob_all)

    log_Q, log_q = IFP.compute_log_Q(
        ifp,
        x_prime,
        log_j=log_j,
        n=n,
        log_q_current=log_q_current,
    )

    assert len(log_Q) == len(x_prime)
    assert log_q.shape == (len(x_prime), n_flows + 1)
    assert all(log_q[:, 0] == 0)
    assert all(log_q[:, -1] == (log_q_current + log_j))

    weights = ifp.poolsize.astype(float)
    weights[-1] = n
    weights /= np.sum(weights)
    expected_log_Q = logsumexp(log_q, b=weights, axis=1)
    np.testing.assert_array_equal(log_Q, expected_log_Q)


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

    ifp.flow.n_models = 15
    ifp.rescale = MagicMock(side_effect=rescale)
    ifp.get_proposal_log_prob = MagicMock(side_effect=get_proposal_log_prob)

    out = IFP.compute_kl_between_proposals(ifp, x, p_it, q_it)

    assert isinstance(out, float)
    assert np.isfinite(out)
