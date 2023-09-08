# -*- coding: utf-8 -*-
"""
Test the analytic proposal method.
"""
import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, create_autospec, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import AnalyticProposal


@pytest.fixture
def proposal():
    return create_autospec(AnalyticProposal)


def test_init(proposal):
    """Test the init method."""
    with patch("nessai.proposal.analytic.Proposal.__init__") as mock_super:
        AnalyticProposal.__init__(proposal, "model", poolsize=10, test=True)
    mock_super.assert_called_once_with("model", test=True)
    assert proposal._poolsize == 10
    assert proposal.populated is False


def test_poolsize(proposal):
    """Test poolsize property"""
    proposal._poolsize = 100
    poolsize = AnalyticProposal.poolsize.__get__(proposal)
    assert poolsize == 100


@pytest.mark.parametrize("N", [None, 5])
def test_populate(proposal, N):
    """Test the populate process"""
    poolsize = 10
    if N is None:
        samples = numpy_array_to_live_points(
            np.arange(poolsize)[:, np.newaxis], "x"
        )
        log_p = np.arange(poolsize, 2 * poolsize)
    else:
        samples = numpy_array_to_live_points(np.arange(N)[:, np.newaxis], "x")
        log_p = np.arange(N, 2 * N)
    log_l = np.random.rand(samples.size)
    proposal.poolsize = poolsize
    proposal.model = Mock()
    proposal.model.new_point = Mock(return_value=samples)
    proposal.model.batch_evaluate_log_prior = Mock(return_value=log_p)
    proposal.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=log_l
    )
    AnalyticProposal.populate(proposal, N=N)

    if N is None:
        N = poolsize
    proposal.model.new_point.assert_called_once_with(N=N)
    proposal.model.batch_evaluate_log_prior.assert_called_once_with(samples)
    proposal.model.batch_evaluate_log_likelihood.assert_called_once_with(
        proposal.samples
    )

    np.testing.assert_array_equal(proposal.samples["logP"], log_p)
    assert sorted(proposal.indices) == list(range(N))
    assert proposal.populated is True
    np.testing.assert_array_equal(proposal.samples["logL"], log_l)


@pytest.mark.parametrize("populated", [True, False])
def test_draw(proposal, populated, wait):
    """Test the draw method"""
    N = 5
    proposal.populated = populated
    proposal.populate = Mock(side_effect=wait)
    proposal.population_time = datetime.timedelta()
    proposal.indices = [0, 1, 2, 3, 4]
    proposal.samples = np.array([0, 1, 2, 3, 4])

    sample = AnalyticProposal.draw(proposal, 1, N=N)

    assert sample == 4

    if not populated:
        proposal.populate.assert_called_once_with(N=N)
        assert proposal.population_time.total_seconds() > 0.0
    else:
        proposal.populate.assert_not_called()
        assert proposal.population_time.total_seconds() == 0.0
    assert proposal.indices == [0, 1, 2, 3]


def test_draw_out_of_samples(proposal):
    """Assert populated is set to false when the last sample is used."""
    N = 5
    proposal.populated = True
    proposal.indices = [0]
    proposal.samples = np.array([0, 1, 2, 3, 4])

    sample = AnalyticProposal.draw(proposal, 1, N=N)

    assert sample == 0
    assert proposal.indices == []
    assert proposal.populated is False


@pytest.mark.integration_test
def test_draw_intergration(model):
    """Integration test for the draw method"""
    proposal = AnalyticProposal(model)
    N = 10
    old_point = model.new_point()
    assert not proposal.populated
    proposal.draw(old_point, N=N)
    assert proposal.populated
    [proposal.draw(old_point) for _ in range(N - 1)]
    assert not proposal.indices
    assert not proposal.populated


@pytest.mark.integration_test
def test_populate_integration(model):
    """Integration test for the populate method"""
    proposal = AnalyticProposal(model)
    N = 500
    proposal.populate(N=N)
    assert proposal.samples.size == N
    assert proposal.populated is True
