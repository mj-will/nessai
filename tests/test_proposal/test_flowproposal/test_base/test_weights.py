"""Test methods related to computing weights"""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from nessai import config
from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal.flowproposal.base import BaseFlowProposal
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture()
def z():
    return np.random.randn(2, 2)


@pytest.fixture()
def x(z):
    return numpy_array_to_live_points(np.random.randn(*z.shape), ["x", "y"])


@pytest.fixture()
def log_q(x):
    return np.random.randn(x.size)


def test_log_prior_wo_reparameterisation(proposal, x):
    """Test the lop prior method"""
    log_prior = -np.ones(x.size)
    proposal._reparameterisation = None
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_prior = MagicMock(return_value=log_prior)

    log_prior_out = BaseFlowProposal.log_prior(proposal, x)

    assert np.array_equal(log_prior, log_prior_out)
    proposal.model.batch_evaluate_log_prior.assert_called_once_with(x)


def test_log_prior_w_reparameterisation(proposal, x):
    """Test the lop prior method with reparameterisations"""
    log_prior = -np.ones(x.size)
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.log_prior = MagicMock(return_value=log_prior)
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_prior = MagicMock(
        return_value=log_prior.copy()
    )

    log_prior_out = BaseFlowProposal.log_prior(proposal, x)

    assert np.array_equal(log_prior_out, -2 * np.ones(x.size))
    proposal._reparameterisation.log_prior.assert_called_once_with(x)
    proposal.model.batch_evaluate_log_prior.assert_called_once_with(x)


def test_prime_log_prior(proposal):
    """Make sure the prime prior raises an error by default."""
    with pytest.raises(RuntimeError) as excinfo:
        BaseFlowProposal.x_prime_log_prior(proposal, 1.0)
    assert "Prime prior is not implemented" in str(excinfo.value)


def test_unit_hypercube_log_prior_wo_reparameterisation(proposal, x):
    log_prior = -np.ones(x.size)
    proposal._reparameterisation = None
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_prior_unit_hypercube = MagicMock(
        return_value=log_prior
    )

    log_prior_out = BaseFlowProposal.unit_hypercube_log_prior(proposal, x)

    assert np.array_equal(log_prior, log_prior_out)
    proposal.model.batch_evaluate_log_prior_unit_hypercube.assert_called_once_with(  # noqa: E501
        x
    )


def test_unit_hypercube_log_prior_w_reparameterisation(proposal, x):
    log_prior = -np.ones(x.size)
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.log_prior = MagicMock(return_value=log_prior)
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_prior_unit_hypercube = MagicMock(
        return_value=log_prior.copy()
    )

    log_prior_out = BaseFlowProposal.unit_hypercube_log_prior(proposal, x)

    assert np.array_equal(log_prior_out, -2 * np.ones(x.size))
    proposal._reparameterisation.log_prior.assert_called_once_with(x)
    proposal.model.batch_evaluate_log_prior_unit_hypercube.assert_called_once_with(  # noqa: E501
        x
    )


@pytest.mark.parametrize(
    "acceptance, scale", [(0.0, 10.0), (0.5, 2.0), (0.01, 10.0), (2.0, 1.0)]
)
def test_update_poolsize_scale(proposal, acceptance, scale):
    """
    Test the check the poolsize is correct adjusted based on the acceptance.
    """
    proposal.max_poolsize_scale = 10.0
    BaseFlowProposal.update_poolsize_scale(proposal, acceptance)
    assert proposal._poolsize_scale == scale


def test_compute_weights(proposal, x, log_q):
    """Test method for computing rejection sampling weights"""
    proposal.use_x_prime_prior = False
    proposal.log_prior = MagicMock(return_value=-np.ones(x.size))
    log_w = BaseFlowProposal.compute_weights(proposal, x, log_q)

    proposal.log_prior.assert_called_once_with(x)
    out = -1 - log_q
    np.testing.assert_array_equal(log_w, out)


def test_compute_weights_return_prior(proposal, x, log_q):
    """Assert prior is returned"""
    proposal.use_x_prime_prior = False
    log_p = -np.ones(x.size)
    proposal.log_prior = MagicMock(return_value=log_p)
    log_w, log_p_out = BaseFlowProposal.compute_weights(
        proposal, x, log_q, return_log_prior=True
    )

    proposal.log_prior.assert_called_once_with(x)
    expected = -1 - log_q
    np.testing.assert_array_equal(log_w, expected)
    assert log_p_out is log_p


def test_compute_weights_prime_prior(proposal, x, log_q):
    """Test method for computing rejection sampling weights with the prime
    prior.
    """
    proposal.use_x_prime_prior = True
    proposal.x_prime_log_prior = MagicMock(return_value=-np.ones(x.size))
    log_w = BaseFlowProposal.compute_weights(proposal, x, log_q)

    proposal.x_prime_log_prior.assert_called_once_with(x)
    out = -1 - log_q
    np.testing.assert_array_equal(log_w, out)


def test_compute_weights_unit_hypercube(proposal, x, log_q):
    proposal.use_x_prime_prior = False
    proposal.map_to_unit_hypercube = True
    proposal.unit_hypercube_log_prior = MagicMock(
        return_value=-np.ones(x.size)
    )
    log_w = BaseFlowProposal.compute_weights(proposal, x, log_q)

    proposal.unit_hypercube_log_prior.assert_called_once_with(x)
    out = -1 - log_q
    np.testing.assert_array_equal(log_w, out)


def test_compute_acceptance(proposal):
    """Test the compute_acceptance method"""
    proposal.samples = np.arange(1, 11, dtype=float).view([("logL", "f8")])
    acc = BaseFlowProposal.compute_acceptance(proposal, 5.0)
    assert acc == 0.5


def test_convert_to_samples(proposal):
    """Test convert to sample without the prime prior"""
    samples = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    proposal.use_x_prime_prior = False
    proposal.model = MagicMock()
    proposal.model.names = ["x"]
    proposal.model.batch_evaluate_log_prior = MagicMock(
        return_value=np.ones(10)
    )

    out_samples = BaseFlowProposal.convert_to_samples(
        proposal, samples, plot=True
    )

    assert out_samples.dtype.names == ("x",) + tuple(
        config.livepoints.non_sampling_parameters
    )


@patch("nessai.proposal.flowproposal.base.plot_1d_comparison")
def test_convert_to_samples_with_prime(mock_plot, proposal):
    """Test convert to sample with the prime prior"""
    samples = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    proposal.use_x_prime_prior = True
    proposal.model = MagicMock()
    proposal.model.names = ["x"]
    proposal.model.batch_evaluate_log_prior = MagicMock(
        return_value=np.ones(10)
    )
    proposal._plot_pool = True
    proposal.training_data_prime = "data"
    proposal.output = os.getcwd()
    proposal.populated_count = 1
    proposal.inverse_rescale = MagicMock(return_value=(samples, None))

    out_samples = BaseFlowProposal.convert_to_samples(
        proposal, samples, plot=True
    )

    mock_plot.assert_called_once_with(
        "data",
        samples,
        labels=["live points", "pool"],
        filename=os.path.join(os.getcwd(), "pool_prime_1.png"),
    )
    proposal.inverse_rescale.assert_called_once()
    assert out_samples.dtype.names == ("x",) + tuple(
        config.livepoints.non_sampling_parameters
    )


def test_convert_to_samples_unit_hypercube(proposal):
    """Test convert to sample without the prime prior"""
    samples = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    samples_hyper = samples.copy()
    samples["x"] /= 2
    proposal.use_x_prime_prior = False
    proposal.map_to_unit_hypercube = True
    proposal.model = MagicMock()
    proposal.model.names = ["x"]
    proposal.model.batch_evaluate_log_prior = MagicMock(
        return_value=np.ones(10)
    )
    proposal.model.from_unit_hypercube = MagicMock(return_value=samples_hyper)

    out_samples = BaseFlowProposal.convert_to_samples(
        proposal, samples, plot=True
    )

    assert out_samples.dtype.names == ("x",) + tuple(
        config.livepoints.non_sampling_parameters
    )
    np.testing.assert_array_equal(out_samples["x"], samples_hyper["x"])


def test_check_prior_bounds(proposal):
    """Test the check prior bounds method."""
    x = numpy_array_to_live_points(np.arange(10)[:, np.newaxis], ["x"])
    y = np.arange(10)
    proposal.model = Mock()
    proposal.model.in_bounds = MagicMock(
        return_value=np.array(6 * [True] + 4 * [False])
    )
    x_out, y_out = BaseFlowProposal.check_prior_bounds(proposal, x, y)

    assert_structured_arrays_equal(x_out, x[:6])
    np.testing.assert_array_equal(y_out, y[:6])
