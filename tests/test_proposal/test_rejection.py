# -*- coding: utf-8 -*-
"""
Test the rejection proposal class.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, create_autospec, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import RejectionProposal
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture
def proposal():
    return create_autospec(RejectionProposal)


def test_init(proposal):
    """Test the init method."""
    with patch(
        "nessai.proposal.rejection.AnalyticProposal.__init__"
    ) as mock_super:
        RejectionProposal.__init__(proposal, "model", poolsize=10, test=True)
    mock_super.assert_called_once_with("model", poolsize=10, test=True)
    assert proposal._checked_population is True
    assert proposal.population_acceptance is None


@pytest.mark.parametrize("N", [None, 5])
def test_draw_proposal(proposal, N):
    """Assert `model.new_point` is called with the corred number of samples."""
    points = np.array([1, 2])
    proposal.poolsize = 10
    proposal.model = Mock()
    proposal.model.new_point = Mock(return_value=points)
    samples = RejectionProposal.draw_proposal(proposal, N=N)
    np.testing.assert_array_equal(samples, points)
    if N is None:
        proposal.model.new_point.assert_called_once_with(N=10)
    else:
        proposal.model.new_point.assert_called_once_with(N=5)


def test_log_proposal(proposal):
    """Assert the correct method from the model is called"""
    x = np.array([3, 4])
    log_prob = np.array([1, 2])
    proposal.model = Mock()
    proposal.model.new_point_log_prob = Mock(return_value=log_prob)
    out = RejectionProposal.log_proposal(proposal, x)
    proposal.model.new_point_log_prob.assert_called_once_with(x)
    np.testing.assert_array_equal(out, log_prob)


def test_compute_weights(proposal):
    """Test the compute weights method"""
    x = numpy_array_to_live_points(np.array([[1], [2], [3]]), "x")
    proposal.model = Mock()
    proposal.model.log_prior = Mock(return_value=np.array([6, 6, 6]))
    proposal.log_proposal = Mock(return_value=np.array([3, 4, np.nan]))
    log_w = np.array([0, -1, np.nan])
    out = RejectionProposal.compute_weights(proposal, x)

    proposal.model.log_prior.assert_called_once_with(x)
    proposal.log_proposal.assert_called_once_with(x)
    np.testing.assert_array_equal(out, log_w)


@pytest.mark.parametrize("N", [None, 4])
def test_populate(proposal, N):
    """Test the populate method"""
    poolsize = 8
    if N is None:
        log_w = np.arange(poolsize)
    else:
        log_w = np.arange(N)
    x = numpy_array_to_live_points(np.random.randn(log_w.size, 1), ["x"])
    u = np.exp(log_w.copy() + 1)
    # These points will have log_u ~ -inf so corresponding samples will be
    # accepted.
    u[::2] = 1e-10
    samples = x[::2]
    log_l = np.log(np.random.rand(samples.size))
    samples["logL"] = log_l
    proposal.poolsize = poolsize
    proposal.populated = False
    proposal.draw_proposal = Mock(return_value=x)
    proposal.compute_weights = Mock(return_value=log_w)
    proposal.model = Mock()
    proposal.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=log_l
    )

    with patch("numpy.random.rand", return_value=u):
        RejectionProposal.populate(proposal, N=N)

    assert proposal.population_acceptance == 0.5
    assert proposal.populated is True
    assert_structured_arrays_equal(proposal.samples, samples)

    if N is None:
        N = poolsize
    proposal.draw_proposal.assert_called_once_with(N=N)

    proposal.model.batch_evaluate_log_likelihood.assert_called_once_with(
        proposal.samples
    )
    assert sorted(proposal.indices) == list(range(samples.size))
    np.testing.assert_array_equal(proposal.samples["logL"], log_l)


@pytest.mark.integration_test
def test_populate_integration(model):
    """Integration test for the populate method"""
    proposal = RejectionProposal(model)
    N = 500
    proposal.populate(N=N)
    assert proposal.samples.size == N
    assert proposal.populated is True
