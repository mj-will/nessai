"""Tests related to sampling from ImportanceFlowProposal"""
from unittest.mock import MagicMock, create_autospec

from nessai.flowmodel.importance import ImportanceFlowModel
from nessai.livepoint import numpy_array_to_live_points, live_points_to_array
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
        return model.from_unit_hypercube(x), np.zeros(x.size)

    def rescale(x):
        x = model.to_unit_hypercube(x)
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
