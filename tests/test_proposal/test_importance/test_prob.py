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
    ifp._weights = {"-1": 0.5, "1": 0.5}
    ifp.weights = ifp._weights
    ifp.log_q_dtype = np.dtype([("-1", "f8"), ("0", "f8"), ("1", "f8")])
    ifp.weights_array = IFP.weights_array.__get__(ifp, IFP)
    weights = {"-1": 1 / 3, "0": 1 / 3, "1": 1 / 3}
    IFP.update_weights(ifp, weights)
    assert ifp._weights == weights


def test_update_proposal_weights_vaild(ifp):
    ifp._weights = {"-1": 0.5, "1": 0.5}
    ifp.weights = ifp._weights
    ifp.log_q_dtype = np.dtype([("-1", "f8"), ("0", "f8"), ("1", "f8")])
    ifp.weights_array = IFP.weights_array.__get__(ifp, IFP)
    weights = {"-1": 0.33, "0": 0.33, "1": 0.33}
    with pytest.raises(RuntimeError, match="Weights must sum to 1!"):
        IFP.update_weights(ifp, weights)


def test_initial_log_prob(ifp):
    x = np.random.randn(10, 2)
    np.testing.assert_array_equal(IFP._log_prob_initial(ifp, x), np.zeros(10))


def test_get_proposal_log_prob_initial(ifp):
    ifp._log_prob_initial = object()
    func = IFP.get_proposal_log_prob(ifp, -1)
    assert func is ifp._log_prob_initial


def test_log_prob_meta_proposal(ifp, x_prime):
    n_flows = 3
    ifp._weights = {"-1": 0.25, "0": 0.25, "1": 0.25, "2": 0.25}
    ifp.weights = ifp._weights
    ifp.weights_array = MagicMock(
        return_value=np.array([0.25, 0.25, 0.25, 0.25])
    )
    ifp.flow.n_models = n_flows
    ifp.flow.models = {
        "0": MagicMock(training=False),
        "1": MagicMock(training=False),
        "2": MagicMock(training=False),
    }
    ifp.log_q_dtype = np.dtype(
        [("-1", "f8"), ("0", "f8"), ("1", "f8"), ("2", "f8")]
    )
    ifp.get_proposal_log_prob = MagicMock(
        side_effect=lambda it, log_j=None: (
            (lambda x: np.zeros(len(x)))
            if it == "-1"
            else (lambda x: np.log(np.random.rand(len(x))))
        )
    )
    ifp.log_prob_meta_proposal_from_log_q = (
        IFP.log_prob_meta_proposal_from_log_q.__get__(ifp, IFP)
    )
    ifp.n_proposals = n_flows + 1

    log_j = np.log(np.random.rand(len(x_prime)))

    def log_prob_ith(x, it):
        return np.log(np.random.rand(len(x)))

    ifp.flow.log_prob_ith = MagicMock(side_effect=log_prob_ith)

    log_Q, log_q = IFP.log_prob_meta_proposal(ifp, x_prime, log_j=log_j)

    assert len(log_Q) == len(x_prime)
    assert log_q.shape == (len(x_prime),)
    assert log_q.dtype.names == ("-1", "0", "1", "2")
    assert np.all(log_q["-1"] == 0)

    log_q_values = np.column_stack([log_q[name] for name in log_q.dtype.names])
    expected_log_Q = logsumexp(
        log_q_values, b=ifp.weights_array(log_q.dtype.names), axis=1
    )
    np.testing.assert_array_equal(log_Q, expected_log_Q)


def test_log_prob_meta_proposal_weights_not_set(ifp, x_prime):
    """Assert an error is raised if a flow is in training mode"""
    n_flows = 3
    ifp._weights = {
        "-1": 0.25,
        "0": 0.25,
        "1": 0.25,
        "2": np.nan,
    }
    ifp.weights = ifp._weights
    ifp.flow.n_models = n_flows
    ifp.log_q_dtype = np.dtype(
        [("-1", "f8"), ("0", "f8"), ("1", "f8"), ("2", "f8")]
    )
    ifp.get_proposal_log_prob = MagicMock(
        side_effect=lambda it, log_j=None: (
            (lambda x: np.zeros(len(x)))
            if it == "-1"
            else (lambda x: np.log(np.random.rand(len(x))))
        )
    )
    ifp.n_proposals = n_flows + 1
    log_j = np.log(np.random.rand(len(x_prime)))
    with pytest.raises(RuntimeError, match="Some weights are not set!"):
        IFP.log_prob_meta_proposal(ifp, x_prime, log_j=log_j)


def test_log_prob_meta_proposal_flow_training(ifp, x_prime):
    """Assert an error is raised if a flow is in training mode"""
    n_flows = 3
    ifp._weights = {"-1": 0.25, "0": 0.25, "1": 0.25, "2": 0.25}
    ifp.weights = ifp._weights
    ifp.weights_array = MagicMock(
        return_value=np.array([0.25, 0.25, 0.25, 0.25])
    )
    ifp.flow.n_models = n_flows
    ifp.flow.models = {
        "0": MagicMock(training=False),
        "1": MagicMock(training=False),
        "2": MagicMock(training=True),
    }
    ifp.log_q_dtype = np.dtype(
        [("-1", "f8"), ("0", "f8"), ("1", "f8"), ("2", "f8")]
    )
    ifp.get_proposal_log_prob = MagicMock(
        side_effect=lambda it, log_j=None: (
            (lambda x: np.zeros(len(x)))
            if it == "-1"
            else (lambda x: np.log(np.random.rand(len(x))))
        )
    )
    ifp.n_proposals = n_flows + 1
    log_j = np.log(np.random.rand(len(x_prime)))
    with pytest.raises(
        RuntimeError, match="One or more flows are in training mode"
    ):
        IFP.log_prob_meta_proposal(ifp, x_prime, log_j=log_j)


@pytest.mark.parametrize("p_it, q_it", [(None, None), (-1, 0), (3, 4)])
def test_kl_between_proposals(ifp, model, p_it, q_it, x):
    def rescale(x):
        x = model.to_unit_hypercube(x)
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def get_proposal_log_prob(it):
        def log_prob(x):
            if it == "-1":
                return np.zeros(len(x))
            else:
                return np.log(np.random.rand(len(x)))

        return log_prob

    ifp.flow.n_models = 15
    ifp.flow.models = {str(i): MagicMock(training=False) for i in range(15)}
    ifp.rescale = MagicMock(side_effect=rescale)
    ifp.get_proposal_log_prob = MagicMock(side_effect=get_proposal_log_prob)

    out = IFP.compute_kl_between_proposals(ifp, x, p_it, q_it)

    assert isinstance(out, float)
    assert np.isfinite(out)


def test_update_log_q(ifp, model, x):
    n_proposals = 5
    ifp._proposal_count = 4
    ifp.proposal_id = "4"

    names = ["-1", "0", "1", "2"]
    log_q = np.empty(len(x), dtype=[(name, "f8") for name in names])
    for name in names:
        log_q[name] = np.log(np.random.rand(len(x)))

    def rescale(x):
        x = model.to_unit_hypercube(x)
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def get_proposal_log_prob(it, log_j=None):
        assert it == "4"

        def log_prob(x):
            return np.log(np.random.rand(len(x)))

        return log_prob

    ifp.n_proposals = n_proposals
    ifp.flow.n_models = n_proposals - 1
    ifp.rescale = MagicMock(side_effect=rescale)
    ifp.get_proposal_log_prob = MagicMock(side_effect=get_proposal_log_prob)

    log_q_out = IFP.update_log_q(ifp, x, log_q)

    assert log_q_out.shape == (len(x),)
    assert log_q_out.dtype.names == ("-1", "0", "1", "2", "4")


def test_compute_meta_proposal_from_log_q(ifp):
    n = 100
    n_prop = 10
    names = [str(i - 1) for i in range(n_prop)]
    log_q = np.empty(n, dtype=[(name, "f8") for name in names])
    for name in names:
        log_q[name] = np.log(np.random.rand(n))

    poolsize = np.random.multinomial(
        n_prop,
        pvals=np.ones(n_prop) / float(n_prop),
        size=n,
    )
    weights = poolsize / np.sum(poolsize)
    ifp.weights_array = MagicMock(return_value=weights)

    log_q_values = np.column_stack([log_q[name] for name in log_q.dtype.names])
    expected = logsumexp(log_q_values, b=weights, axis=1)

    out = IFP.log_prob_meta_proposal_from_log_q(ifp, log_q)

    assert len(out) == len(log_q)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.usefixtures("ins_parameters")
def test_compute_meta_proposal_samples(ifp, x, x_prime, log_j):
    ifp._proposal_count = 2
    ifp._weights = {"-1": 0.25, "0": 0.25, "1": 0.25, "2": 0.25}
    ifp.weights = ifp._weights
    ifp.proposal_id = "2"

    x["logQ"] = np.nan
    x["logW"] = np.nan

    log_Q = np.log(np.random.rand(len(x)))
    log_q = np.empty(len(x), dtype=[(str(i), "f8") for i in range(10)])
    for name in log_q.dtype.names:
        log_q[name] = np.log(np.random.rand(len(x)))

    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.log_prob_meta_proposal = MagicMock(return_value=(log_Q, log_q))

    log_Q_out, log_q_out = IFP.compute_meta_proposal_samples(ifp, x)

    ifp.rescale.assert_called_once_with(x)
    ifp.log_prob_meta_proposal.assert_called_once_with(x_prime, log_j=log_j)

    np.testing.assert_array_equal(log_Q_out, log_Q)
    np.testing.assert_array_equal(log_q_out, log_q)


@pytest.mark.parametrize(
    "weights", [{"-1": 0.5, "0": 0.5}, {"-1": 0.5, "0": 0.5, "1": np.nan}]
)
@pytest.mark.usefixtures("ins_parameters")
def test_compute_meta_proposal_samples_weights_error(ifp, x, weights):
    ifp._proposal_count = 1
    ifp._weights = weights
    with pytest.raises(RuntimeError, match=r"Weight\(s\) missing or not set."):
        IFP.compute_meta_proposal_samples(ifp, x)
