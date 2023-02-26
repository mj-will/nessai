# -*- coding: utf-8 -*-
"""Test methods related to popluation of the proposal after training"""
from functools import partial
import os

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch, call

from nessai import config
from nessai.proposal import FlowProposal
from nessai.livepoint import get_dtype, numpy_array_to_live_points
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
    proposal.model.log_prior = MagicMock(return_value=log_prior)

    log_prior_out = FlowProposal.log_prior(proposal, x)

    assert np.array_equal(log_prior, log_prior_out)
    proposal.model.log_prior.assert_called_once_with(x)


def test_log_prior_w_reparameterisation(proposal, x):
    """Test the lop prior method with reparameterisations"""
    log_prior = -np.ones(x.size)
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.log_prior = MagicMock(return_value=log_prior)
    proposal.model = MagicMock()
    proposal.model.log_prior = MagicMock(return_value=log_prior)

    log_prior_out = FlowProposal.log_prior(proposal, x)

    assert np.array_equal(log_prior_out, -2 * np.ones(x.size))
    proposal._reparameterisation.log_prior.assert_called_once_with(x)
    proposal.model.log_prior.assert_called_once_with(x)


def test_prime_log_prior(proposal):
    """Make sure the prime prior raises an error by default."""
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.x_prime_log_prior(proposal, 1.0)
    assert "Prime prior is not implemented" in str(excinfo.value)


@pytest.mark.parametrize(
    "acceptance, scale", [(0.0, 10.0), (0.5, 2.0), (0.01, 10.0), (2.0, 1.0)]
)
def test_update_poolsize_scale(proposal, acceptance, scale):
    """
    Test the check the poolsize is correct adjusted based on the acceptance.
    """
    proposal.max_poolsize_scale = 10.0
    FlowProposal.update_poolsize_scale(proposal, acceptance)
    assert proposal._poolsize_scale == scale


def test_compute_weights(proposal, x, log_q):
    """Test method for computing rejection sampling weights"""
    proposal.use_x_prime_prior = False
    proposal.log_prior = MagicMock(return_value=-np.ones(x.size))
    log_w = FlowProposal.compute_weights(proposal, x, log_q)

    proposal.log_prior.assert_called_once_with(x)
    out = -1 - log_q
    out -= out.max()
    assert np.array_equal(log_w, out)


def test_compute_weights_prime_prior(proposal, x, log_q):
    """Test method for computing rejection sampling weights with the prime
    prior.
    """
    proposal.use_x_prime_prior = True
    proposal.x_prime_log_prior = MagicMock(return_value=-np.ones(x.size))
    log_w = FlowProposal.compute_weights(proposal, x, log_q)

    proposal.x_prime_log_prior.assert_called_once_with(x)
    out = -1 - log_q
    out -= out.max()
    assert np.array_equal(log_w, out)


@patch("numpy.random.rand", return_value=np.array([0.1, 0.9]))
def test_rejection_sampling(proposal, z, x, log_q):
    """Test rejection sampling method."""
    proposal.use_x_prime_prior = False
    proposal.truncate = False
    proposal.backward_pass = MagicMock(return_value=(x, log_q))
    log_w = np.log(np.array([0.5, 0.5]))
    proposal.compute_weights = MagicMock(return_value=log_w)

    z_out, x_out = FlowProposal.rejection_sampling(proposal, z)

    assert proposal.backward_pass.called_once_with(x, True)
    assert proposal.compute_weights.called_once_with(x)
    assert x_out.size == 1
    assert z_out.shape == (1, 2)
    assert_structured_arrays_equal(x_out[0], x[0])
    assert np.array_equal(z_out[0], z[0])


def test_rejection_sampling_empty(proposal, z):
    """Test rejection sampling method if no valid points are produced by
    `backwards_pass`
    """
    proposal.use_x_prime_prior = False
    proposal.truncate = False
    proposal.backward_pass = MagicMock(
        return_value=(np.array([]), np.array([]))
    )

    z_out, x_out = FlowProposal.rejection_sampling(proposal, z)

    assert x_out.size == 0
    assert z_out.size == 0


@patch("numpy.random.rand", return_value=np.array([0.1]))
def test_rejection_sampling_truncate(proposal, z, x):
    """Test rejection sampling method with truncation"""
    proposal.use_x_prime_prior = False
    proposal.truncate = True
    log_q = np.array([0.0, 1.0])
    proposal.backward_pass = MagicMock(return_value=(x, log_q))
    worst_q = 0.5
    log_w = np.log(np.array([0.5]))
    proposal.compute_weights = MagicMock(return_value=log_w)

    z_out, x_out = FlowProposal.rejection_sampling(
        proposal, z, worst_q=worst_q
    )

    assert proposal.backward_pass.called_once_with(x, True)
    assert proposal.compute_weights.called_once_with(x)
    assert x_out.size == 1
    assert z_out.shape == (1, 2)
    assert_structured_arrays_equal(x_out[0], x[1])
    assert np.array_equal(z_out[0], z[1])


def test_rejection_sampling_truncate_missing_q(proposal, z, x, log_q):
    """Test rejection sampling method with truncation without without q"""
    proposal.use_x_prime_prior = False
    proposal.truncate = True
    log_q = np.array([0.0, 1.0])
    proposal.backward_pass = MagicMock(return_value=(x, log_q))

    with pytest.raises(ValueError) as excinfo:
        FlowProposal.rejection_sampling(proposal, z, worst_q=None)
    assert "`worst_q` is None but truncation is enabled" in str(excinfo.value)


def test_compute_acceptance(proposal):
    """Test the compute_acceptance method"""
    proposal.samples = np.arange(1, 11, dtype=float).view([("logL", "f8")])
    acc = FlowProposal.compute_acceptance(proposal, 5.0)
    assert acc == 0.5


def test_convert_to_samples(proposal):
    """Test convert to sample without the prime prior"""
    samples = numpy_array_to_live_points(np.random.randn(10, 4), ["x", "y"])
    proposal.use_x_prime_prior = False
    proposal.model = MagicMock()
    proposal.model.names = ["x"]
    proposal.model.log_prior = MagicMock(return_value=np.ones(10))

    out_samples = FlowProposal.convert_to_samples(proposal, samples, plot=True)

    assert out_samples.dtype.names == ("x",) + tuple(
        config.livepoints.non_sampling_parameters
    )


@patch("nessai.proposal.flowproposal.plot_1d_comparison")
def test_convert_to_samples_with_prime(mock_plot, proposal):
    """Test convert to sample with the prime prior"""
    samples = numpy_array_to_live_points(np.random.randn(10, 4), ["x", "y"])
    proposal.use_x_prime_prior = True
    proposal.model = MagicMock()
    proposal.model.names = ["x"]
    proposal.model.log_prior = MagicMock(return_value=np.ones(10))
    proposal._plot_pool = True
    proposal.training_data_prime = "data"
    proposal.output = os.getcwd()
    proposal.populated_count = 1
    proposal.inverse_rescale = MagicMock(return_value=(samples, None))

    out_samples = FlowProposal.convert_to_samples(proposal, samples, plot=True)

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


def test_get_alt_distribution_truncated_gaussian(proposal):
    """
    Test getting the alternative distribution for the default latent prior, the
    truncated Gaussian with var=1. This should return None.
    """
    proposal.draw_latent_kwargs = {}
    proposal.latent_prior = "truncated_gaussian"
    dist = FlowProposal.get_alt_distribution(proposal)
    assert dist is None


@pytest.mark.parametrize("prior", ["uniform_nsphere", "uniform_nball"])
def test_get_alt_distribution_uniform(proposal, prior):
    """
    Test getting the alternative distribution for priors that are uniform in
    the n-ball.
    """
    proposal.latent_prior = prior
    proposal.dims = 2
    proposal.r = 2.0
    proposal.fuzz = 1.2
    proposal.flow = Mock()
    proposal.flow.device = "cpu"
    with patch(
        "nessai.proposal.flowproposal.get_uniform_distribution"
    ) as mock:
        dist = FlowProposal.get_alt_distribution(proposal)

    assert dist is not None
    mock.assert_called_once_with(2, 2.4, device="cpu")


def test_radius(proposal):
    """Test computing the radius"""
    z = np.array([[1, 2, 3], [0, 1, 2]])
    expected_r = np.sqrt(14)
    r = FlowProposal.radius(proposal, z)
    assert r == expected_r


def test_radius_w_log_q(proposal):
    """Test computing the radius with log_q"""
    z = np.array([[1, 2, 3], [0, 1, 2]])
    log_q = np.array([1, 2])
    expected_r = np.sqrt(14)
    r, log_q_r = FlowProposal.radius(proposal, z, log_q=log_q)
    assert r == expected_r
    assert log_q_r == log_q[0]


def test_check_prior_bounds(proposal):
    """Test the check prior bounds method."""
    x = numpy_array_to_live_points(np.arange(10)[:, np.newaxis], ["x"])
    y = np.arange(10)
    proposal.model = Mock()
    proposal.model.in_bounds = MagicMock(
        return_value=np.array(6 * [True] + 4 * [False])
    )
    x_out, y_out = FlowProposal.check_prior_bounds(proposal, x, y)

    assert_structured_arrays_equal(x_out, x[:6])
    np.testing.assert_array_equal(y_out, y[:6])


def test_prep_latent_prior_truncated(proposal):
    """Assert prep latent prior calls the correct values"""

    proposal.latent_prior = "truncated_gaussian"
    proposal.dims = 2
    proposal.r = 3.0
    proposal.fuzz = 1.2
    dist = MagicMock()
    dist.sample = MagicMock()

    with patch(
        "nessai.proposal.flowproposal.NDimensionalTruncatedGaussian",
        return_value=dist,
    ) as mock_dist:
        FlowProposal.prep_latent_prior(proposal)

    mock_dist.assert_called_once_with(2, 3.0, fuzz=1.2)

    assert proposal._populate_dist is dist
    assert proposal._draw_func is dist.sample


def test_prep_latent_prior_other(proposal):
    """Assert partial acts as expected"""
    proposal.latent_prior = "gaussian"
    proposal.dims = 2
    proposal.r = 3.0
    proposal.fuzz = 1.2

    def draw(dims, N=None, r=None, fuzz=None):
        return np.zeros((N, dims))

    proposal._draw_latent_prior = draw

    with patch(
        "nessai.proposal.flowproposal.partial", side_effect=partial
    ) as mock_partial:
        FlowProposal.prep_latent_prior(proposal)

    mock_partial.assert_called_once_with(draw, dims=2, r=3.0, fuzz=1.2)

    assert proposal._draw_func(N=10).shape == (10, 2)


def test_draw_latent_prior(proposal):
    proposal._draw_func = MagicMock(return_value=[1, 2])
    out = FlowProposal.draw_latent_prior(proposal, 2)
    proposal._draw_func.assert_called_once_with(N=2)
    assert out == [1, 2]


@pytest.mark.parametrize("check_acceptance", [False, True])
@pytest.mark.parametrize("indices", [[], [1]])
@pytest.mark.parametrize("r", [None, 1.0])
@pytest.mark.parametrize(
    "max_radius, min_radius",
    [(0.5, None), (None, 1.5), (None, None)],
)
def test_populate(
    proposal, check_acceptance, indices, r, min_radius, max_radius
):
    """Test the main populate method"""
    n_dims = 2
    poolsize = 10
    drawsize = 5
    names = ["x", "y"]
    worst_point = np.array(
        [[1, 2, 3]], dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")]
    )
    worst_z = np.random.randn(1, n_dims)
    worst_q = np.random.randn(1) if r is None else None
    z = [
        np.random.randn(drawsize, n_dims),
        np.random.randn(drawsize, n_dims),
        np.random.randn(drawsize, n_dims),
    ]
    x = [
        numpy_array_to_live_points(np.random.randn(drawsize, n_dims), names),
        numpy_array_to_live_points(np.random.randn(drawsize, n_dims), names),
        numpy_array_to_live_points(np.random.randn(drawsize, n_dims), names),
    ]
    log_l = np.random.rand(poolsize)

    r_flow = 1.0

    if r is None:
        r_out = r_flow
        if min_radius is not None:
            r_out = max(r_out, min_radius)
        if max_radius is not None:
            r_out = min(r_out, max_radius)
    else:
        r_out = r

    proposal.initialised = True
    proposal.max_radius = max_radius
    proposal.dims = n_dims
    proposal.poolsize = poolsize
    proposal.drawsize = drawsize
    proposal.min_radius = min_radius
    proposal.fuzz = 1.0
    proposal.indices = indices
    proposal.acceptance = [0.7]
    proposal.keep_samples = False
    proposal.fixed_radius = False
    proposal.compute_radius_with_all = False
    proposal.check_acceptance = check_acceptance
    proposal._plot_pool = True
    proposal.populated_count = 1
    proposal.population_dtype = get_dtype(["x_prime", "y_prime"])

    proposal.forward_pass = MagicMock(return_value=(worst_z, worst_q))
    proposal.radius = MagicMock(return_value=(r_flow, worst_q))
    proposal.get_alt_distribution = MagicMock(return_value=None)
    proposal.prep_latent_prior = MagicMock()
    proposal.draw_latent_prior = MagicMock(side_effect=z)
    proposal.rejection_sampling = MagicMock(
        side_effect=[(a[:-1], b[:-1]) for a, b in zip(z, x)]
    )
    proposal.compute_acceptance = MagicMock(return_value=0.8)
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=log_l
    )

    proposal.plot_pool = MagicMock()
    proposal.convert_to_samples = MagicMock(
        side_effect=lambda *args, **kwargs: args[0]
    )

    x_empty = np.empty(poolsize, dtype=proposal.population_dtype)
    with patch(
        "nessai.proposal.flowproposal.empty_structured_array",
        return_value=x_empty,
    ) as mock_empty:
        FlowProposal.populate(proposal, worst_point, N=10, plot=True, r=r)

    mock_empty.assert_called_once_with(
        poolsize,
        dtype=proposal.population_dtype,
    )

    if r is None:
        proposal.forward_pass.assert_called_once_with(
            worst_point,
            rescale=True,
            compute_radius=True,
        )
        proposal.radius.assert_called_once_with(worst_z, worst_q)
    else:
        assert proposal.r is r

    assert proposal.r == r_out

    proposal.prep_latent_prior.assert_called_once()

    draw_calls = 3 * [call(5)]
    proposal.draw_latent_prior.assert_has_calls(draw_calls)

    rejection_calls = [
        call(z[0], worst_q),
        call(z[1], worst_q),
        call(z[2], worst_q),
    ]
    proposal.rejection_sampling.assert_has_calls(rejection_calls)

    proposal.plot_pool.assert_called_once()
    proposal.convert_to_samples.assert_called_once()
    assert_structured_arrays_equal(
        proposal.convert_to_samples.call_args[0][0], proposal.x
    )
    assert proposal.convert_to_samples.call_args[1]["plot"] is True

    assert proposal.population_acceptance == (10 / 15)
    assert proposal.populated_count == 2
    assert proposal.populated is True
    assert proposal.x.size == 10

    if check_acceptance:
        proposal.compute_acceptance.assert_called()
        assert proposal.acceptance == [0.7, 0.8]
    else:
        proposal.compute_acceptance.assert_not_called()

    proposal.model.batch_evaluate_log_likelihood.assert_called_once_with(
        proposal.samples
    )
    np.testing.assert_array_equal(proposal.samples["logL"], log_l)


def test_populate_not_initialised(proposal):
    """Assert populate fails if the proposal is not initialised"""
    proposal.initialised = False
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.populate(proposal, 1.0)
    assert "Proposal has not been initialised. " in str(excinfo.value)
