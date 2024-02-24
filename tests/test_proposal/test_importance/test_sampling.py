"""Tests related to sampling from ImportanceFlowProposal"""

from unittest.mock import MagicMock, create_autospec

from nessai.flowmodel.importance import ImportanceFlowModel
from nessai.livepoint import numpy_array_to_live_points, live_points_to_array
from nessai.proposal.importance import ImportanceFlowProposal as IFP
import numpy as np
from scipy.special import logsumexp
import pytest


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.usefixtures("ins_parameters")
def test_draw_from_prior(ifp, x, n):
    """Test drawing from the prior"""
    n_proposals = 4
    x_prime = np.random.randn(n, 2)
    log_j = np.random.rand(n)
    log_Q = np.random.randn(n)
    log_q = np.random.randn(n_proposals, n)
    ifp.model.sample_unit_hypercube = MagicMock(return_value=x)
    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.compute_log_Q = MagicMock(return_value=(log_Q, log_q))

    x_out, log_q_out = IFP.draw_from_prior(ifp, n)
    assert log_q_out.shape == (n_proposals, n)

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

    def log_prob_all(x):
        return np.log(np.random.rand(len(x), n_flows))

    ifp.model = model
    ifp.flow = create_autospec(ImportanceFlowModel)
    ifp.flow.sample_ith = MagicMock(side_effect=sample_ith)
    ifp.flow.log_prob_all = MagicMock(side_effect=log_prob_all)
    ifp.to_prime = MagicMock(side_effect=to_prime)
    ifp.inverse_rescale = MagicMock(side_effect=inverse_rescale)
    ifp.rescale = MagicMock(side_effect=rescale)

    x, log_q, actual_counts = IFP.draw_from_flows(
        ifp, n, weights=weights, counts=counts
    )
    assert len(x) == n
    assert log_q.shape == (n, n_flows + 1)
    assert all(log_q[:, 0] == 0)

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
    ifp.level_count = n_proposals - 1
    ifp.model = model
    ifp.dtype = model.new_point().dtype
    ifp.n_requested = {-1: 100, 0: 100, 2: 100, 3: 200, 4: 0}
    ifp.n_draws = ifp.n_requested.copy()
    ifp.draw_count = n_proposals - 1

    def inverse_rescale(x):
        x = numpy_array_to_live_points(x, model.names)
        return model.from_unit_hypercube(x), np.zeros(x.size)

    def rescale(x):
        x = model.to_unit_hypercube(x)
        x = live_points_to_array(x, model.names)
        return x, np.zeros(x.shape[0])

    def sample_ith(i, N):
        assert i == (n_proposals - 1)
        return np.random.rand(N, model.dims)

    def compute_log_Q(x_prime, log_q_current=None, log_j=None, n=None):
        assert log_q_current is None
        log_q = (
            np.log(np.random.rand(len(x_prime), n_proposals)) + log_j[:, None]
        )
        weights = list(ifp.n_requested.values())
        weights[-1] = n
        assert n == n_draw
        weights /= np.sum(weights)
        log_Q = logsumexp(log_q, b=weights, axis=1)
        return log_Q, log_q

    ifp.rescale = rescale
    ifp.inverse_rescale = inverse_rescale
    ifp.compute_log_Q = compute_log_Q
    ifp.flow = create_autospec(ImportanceFlowModel)
    ifp.flow.sample_ith = sample_ith

    samples_out, log_q_out = IFP.draw(ifp, n_draw)

    assert len(samples_out) == n_draw
    assert log_q_out.shape == (n_draw, n_proposals)
    assert ifp.n_requested[4] == n_draw
    assert ifp.draw_count == n_proposals
