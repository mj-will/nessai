"""Tests related to sampling from ImportanceFlowProposal"""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
from scipy.special import logsumexp

from nessai.flowmodel.importance import ImportanceFlowModel
from nessai.livepoint import live_points_to_array, numpy_array_to_live_points
from nessai.proposal.importance import ImportanceFlowProposal as IFP


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.usefixtures("ins_parameters")
def test_draw_from_prior(ifp, n, model):
    """Test drawing from the prior"""
    n_proposals = 4
    x_prime = np.random.randn(n, 2)
    x = numpy_array_to_live_points(np.random.rand(n, 2), names=model.names)
    log_j = np.random.rand(n)
    log_Q = np.random.randn(n)
    log_q = np.empty(n, dtype=[(str(i - 1), "f8") for i in range(n_proposals)])
    for name in log_q.dtype.names:
        log_q[name] = np.random.randn(n)
    ifp.model.sample_unit_hypercube = MagicMock(return_value=x)
    ifp.model.batch_evaluate_log_prior_unit_hypercube = MagicMock(
        return_value=np.zeros(n)
    )
    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.log_prob_meta_proposal = MagicMock(return_value=(log_Q, log_q))

    x_out, log_q_out = IFP.draw_from_prior(ifp, n)
    assert log_q_out.shape == (n,)
    assert log_q_out.dtype.names == tuple(
        str(i - 1) for i in range(n_proposals)
    )

    assert x_out is x


@pytest.mark.parametrize("n", [1, 3, 10, 100])
@pytest.mark.parametrize("test_counts", [False, True])
def test_draw_from_flows(ifp, model, n, test_counts):
    n_flows = 5
    weights = np.random.rand(n_flows + 1)
    if test_counts:
        counts = np.random.multinomial(n, weights / weights.sum())
        print(counts)
    else:
        counts = None
    ifp.n_proposals = n_flows + 1

    def sample_ith(it, N):
        return np.random.rand(N, model.dims)

    def to_prime(x):
        return x.copy(), np.zeros(x.size)

    def inverse_rescale(x):
        x = numpy_array_to_live_points(x, model.names)
        return x, np.zeros(x.size)

    def rescale(x):
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def log_prob_ith(x, it):
        return np.log(np.random.rand(len(x)))

    ifp.model = model
    ifp.model.in_unit_hypercube = MagicMock(
        side_effect=lambda s: np.ones(s.size, dtype=bool)
    )
    ifp.flow = create_autospec(ImportanceFlowModel)
    ifp.flow.sample_ith = MagicMock(side_effect=sample_ith)
    ifp.flow.log_prob_ith = MagicMock(side_effect=log_prob_ith)
    ifp.to_prime = MagicMock(side_effect=to_prime)
    ifp.inverse_rescale = MagicMock(side_effect=inverse_rescale)
    ifp.rescale = MagicMock(side_effect=rescale)
    ifp._weights = {
        str(i - 1): 1.0 / (n_flows + 1) for i in range(n_flows + 1)
    }
    ifp.log_q_dtype = np.dtype(
        [(str(i - 1), "f8") for i in range(n_flows + 1)]
    )
    ifp.get_proposal_log_prob = MagicMock(
        side_effect=lambda it, log_j=None: (
            (lambda x: np.zeros(len(x)))
            if it == "-1"
            else (lambda x: np.log(np.random.rand(len(x))))
        )
    )

    x, log_q, actual_counts = IFP.draw_from_flows(
        ifp, n, weights=weights, counts=counts
    )
    assert len(x) == n
    assert log_q.shape == (n,)
    assert log_q.dtype.names == tuple(str(i - 1) for i in range(n_flows + 1))
    assert np.all(log_q["-1"] == 0)

    assert np.isfinite(x["logP"]).all()
    assert np.isnan(x["logL"]).all()

    assert np.sum(actual_counts) == n
    if test_counts:
        np.testing.assert_array_equal(actual_counts, counts)


@pytest.mark.usefixtures("ins_parameters")
def test_draw(ifp, model):
    n_proposals = 5
    n_draw = 100
    ifp.n_proposals = n_proposals
    ifp._proposal_count = n_proposals - 1
    ifp.proposal_id = str(n_proposals - 1)
    ifp.model = model
    ifp.dtype = model.new_point().dtype
    ifp._weights = {"-1": 0.2, "0": 0.2, "2": 0.2, "3": 0.4, "4": np.nan}
    ifp.log_q_dtype = np.dtype(
        [(str(i - 1), "f8") for i in range(n_proposals)]
    )

    def inverse_rescale(x):
        x = numpy_array_to_live_points(x, model.names)
        return x, np.zeros(x.size)

    def rescale(x):
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def sample_ith(i, N):
        assert i == str(n_proposals - 1)
        return np.random.rand(N, model.dims)

    def log_prob_meta_proposal(x_prime, log_j=None):
        names = [str(i - 1) for i in range(n_proposals)]
        log_q = np.empty(len(x_prime), dtype=[(n, "f8") for n in names])
        for name in names:
            log_q[name] = np.log(np.random.rand(len(x_prime)))
        log_q_values = np.column_stack([log_q[name] for name in names])
        weights = np.fromiter(ifp._weights.values(), float)
        log_Q = logsumexp(log_q_values, b=weights, axis=1)
        return log_Q, log_q

    ifp.rescale = rescale
    ifp.inverse_rescale = inverse_rescale
    ifp.log_prob_meta_proposal = log_prob_meta_proposal
    ifp.flow = create_autospec(ImportanceFlowModel)
    ifp.flow.sample_ith = sample_ith
    ifp.model.in_unit_hypercube = MagicMock(
        side_effect=lambda s: np.ones(len(s), dtype=bool)
    )
    ifp.model.batch_evaluate_log_prior = MagicMock(
        side_effect=lambda s, unit_hypercube=True: np.zeros(len(s))
    )
    ifp.model.batch_evaluate_log_prior_unit_hypercube = MagicMock(
        side_effect=lambda s: np.zeros(len(s))
    )
    ifp.qid_dtype = np.dtype("U8")
    ifp.cast_qid = IFP.cast_qid.__get__(ifp, IFP)

    samples_out, log_q_out = IFP.draw(ifp, n_draw)

    assert len(samples_out) == n_draw
    assert log_q_out.shape == (n_draw,)
