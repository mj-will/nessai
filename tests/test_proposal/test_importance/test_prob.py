"""Tests methods for computing log_prob etc"""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
from scipy.special import logsumexp

from nessai.flowmodel.importance import ImportanceFlowModel
from nessai.livepoint import live_points_to_array
from nessai.proposal.importance import ImportanceFlowProposal as IFP


@pytest.fixture()
def ifp(ifp):
    ifp.flow = create_autospec(ImportanceFlowModel)
    return ifp


def test_update_proposal_weights(ifp):
    ifp._weights = {-1: 0.5, 1: 0.5}
    weights = {-1: 1 / 3, 0: 1 / 3, 1: 1 / 3}
    IFP.update_proposal_weights(ifp, weights)
    assert ifp._weights == weights


def test_update_proposal_weights_vaild(ifp):
    ifp._weights = {-1: 0.5, 1: 0.5}
    weights = {-1: 0.33, 0: 0.33, 1: 0.33}
    with pytest.raises(RuntimeError, match="Weights must sum to 1!"):
        IFP.update_proposal_weights(ifp, weights)


def test_initial_log_prob(ifp):
    x = np.random.randn(10, 2)
    np.testing.assert_array_equal(IFP._log_prob_initial(ifp, x), np.zeros(10))


def test_get_proposal_log_prob_initial(ifp):
    ifp._log_prob_initial = object()
    func = IFP.get_proposal_log_prob(ifp, -1)
    assert func is ifp._log_prob_initial


def test_compute_log_Q(ifp, x_prime):
    n_flows = 3
    ifp.weights_array = np.array([0.25, 0.25, 0.25, 0.25])
    ifp.flow.n_models = n_flows
    ifp.n_proposals = n_flows + 1

    log_j = np.log(np.random.rand(len(x_prime)))

    def log_prob_all(x):
        return np.log(np.random.rand(len(x), n_flows))

    ifp.flow.log_prob_all = MagicMock(side_effect=log_prob_all)

    log_Q, log_q = IFP.compute_log_Q(ifp, x_prime, log_j=log_j)

    assert len(log_Q) == len(x_prime)
    assert log_q.shape == (len(x_prime), n_flows + 1)
    assert all(log_q[:, 0] == 0)

    expected_log_Q = logsumexp(log_q, b=ifp.weights_array, axis=1)
    np.testing.assert_array_equal(log_Q, expected_log_Q)


def test_compute_log_Q_weights_not_set(ifp, x_prime):
    """Assert an error is raised if a flow is in training mode"""
    n_flows = 3
    ifp.weights = {i - 1: v for i, v in enumerate([0.25, 0.25, 0.25, np.nan])}
    ifp.flow.n_models = n_flows
    ifp.n_proposals = n_flows + 1
    log_j = np.log(np.random.rand(len(x_prime)))
    with pytest.raises(RuntimeError, match="Some weights are not set!"):
        IFP.compute_log_Q(ifp, x_prime, log_j=log_j)


def test_compute_log_Q_flow_training(ifp, x_prime):
    """Assert an error is raised if a flow is in training mode"""
    n_flows = 3
    ifp.weights_array = np.array([0.25, 0.25, 0.25, 0.25])
    ifp.flow.n_models = n_flows
    ifp.flow.models = []
    for _ in range(n_flows):
        mock_model = MagicMock()
        mock_model.training = False
        ifp.flow.models.append(mock_model)
    ifp.flow.models[-1].training = True
    ifp.n_proposals = n_flows + 1
    log_j = np.log(np.random.rand(len(x_prime)))
    with pytest.raises(
        RuntimeError, match="One or more flows are in training mode"
    ):
        IFP.compute_log_Q(ifp, x_prime, log_j=log_j)


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


def test_update_log_q(ifp, model, x):
    n_proposals = 5
    ifp.level_count = 4

    log_q = np.log(np.random.rand(len(x), n_proposals - 1))

    def rescale(x):
        x = model.to_unit_hypercube(x)
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def get_proposal_log_prob(it):
        assert it == 4

        def log_prob(x):
            return np.log(np.random.rand(len(x)))

        return log_prob

    ifp.n_proposals = n_proposals
    ifp.flow.n_models = n_proposals - 1
    ifp.rescale = MagicMock(side_effect=rescale)
    ifp.get_proposal_log_prob = MagicMock(side_effect=get_proposal_log_prob)

    log_q_out = IFP.update_log_q(ifp, x, log_q)

    assert log_q_out.shape == (len(x), n_proposals)


def test_compute_meta_proposal_from_log_q(ifp):
    n = 100
    n_prop = 10
    log_q = np.log(np.random.rand(n, n_prop))

    poolsize = np.random.multinomial(
        n_prop,
        pvals=np.ones(n_prop) / float(n_prop),
        size=n,
    )
    weights = poolsize / np.sum(poolsize)
    ifp.weights_array = weights

    expected = logsumexp(
        log_q,
        b=weights,
        axis=1,
    )

    out = IFP.compute_meta_proposal_from_log_q(ifp, log_q)

    assert len(out) == len(log_q)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.usefixtures("ins_parameters")
def test_compute_meta_proposal_samples(ifp, x, x_prime, log_j):
    ifp.level_count = 2
    ifp.weights = {-1: 0.25, 0: 0.25, 1: 0.25, 2: 0.25}

    x["logQ"] = np.nan
    x["logW"] = np.nan

    log_Q = np.log(np.random.rand(len(x)))
    log_q = np.log(np.random.rand(len(x), 10))

    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.compute_log_Q = MagicMock(return_value=(log_Q, log_q))

    log_Q_out, log_q_out = IFP.compute_meta_proposal_samples(ifp, x)

    ifp.rescale.assert_called_once_with(x)
    ifp.compute_log_Q.assert_called_once_with(x_prime, log_j=log_j)

    np.testing.assert_array_equal(log_Q_out, log_Q)
    np.testing.assert_array_equal(log_q_out, log_q)


@pytest.mark.parametrize(
    "weights", [{-1: 0.5, 0: 0.5}, {-1: 0.5, 0: 0.5, 1: np.nan}]
)
@pytest.mark.usefixtures("ins_parameters")
def test_compute_meta_proposal_samples_weights_error(ifp, x, weights):
    ifp.level_count = 1
    ifp.weights = weights
    with pytest.raises(RuntimeError, match=r"Weight\(s\) missing or not set."):
        IFP.compute_meta_proposal_samples(ifp, x)
