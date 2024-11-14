# -*- coding: utf-8 -*-
"""Test methods related to popluation of the proposal after training"""

import datetime
from functools import partial
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from nessai.livepoint import get_dtype, numpy_array_to_live_points
from nessai.proposal import FlowProposal
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture()
def z(rng):
    return rng.standard_normal((2, 2))


@pytest.fixture()
def x(z, rng):
    return numpy_array_to_live_points(rng.standard_normal(z.shape), ["x", "y"])


@pytest.fixture()
def log_q(x, rng):
    return rng.standard_normal(x.size)


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
        "nessai.proposal.flowproposal.flowproposal.get_uniform_distribution"
    ) as mock:
        dist = FlowProposal.get_alt_distribution(proposal)

    assert dist is not None
    mock.assert_called_once_with(2, 2.4, device="cpu")


def test_radius(proposal):
    """Test computing the radius"""
    z = np.array([[1, 2, 3], [0, 1, 2]])
    expected_r = np.sqrt(14)
    r = FlowProposal.radius(proposal, z)
    np.testing.assert_equal(r, expected_r)


def test_radius_w_log_q(proposal):
    """Test computing the radius with log_q"""
    z = np.array([[1, 2, 3], [0, 1, 2]])
    log_q = np.array([1, 2])
    expected_r = np.sqrt(14)
    r, log_q_r = FlowProposal.radius(proposal, z, log_q)
    assert r == expected_r
    assert log_q_r == log_q[0]


def test_prep_latent_prior_truncated(proposal):
    """Assert prep latent prior calls the correct values"""

    proposal.latent_prior = "truncated_gaussian"
    proposal.dims = 2
    proposal.r = 3.0
    proposal.fuzz = 1.2
    dist = MagicMock()
    dist.sample = MagicMock()

    with patch(
        "nessai.proposal.flowproposal.flowproposal.NDimensionalTruncatedGaussian",  # noqa: E501
        return_value=dist,
    ) as mock_dist:
        FlowProposal.prep_latent_prior(proposal)

    mock_dist.assert_called_once_with(2, 3.0, fuzz=1.2, rng=proposal.rng)

    assert proposal._populate_dist is dist
    assert proposal._draw_func is dist.sample


def test_prep_latent_prior_other(proposal):
    """Assert partial acts as expected"""
    proposal.latent_prior = "gaussian"
    proposal.dims = 2
    proposal.r = 3.0
    proposal.fuzz = 1.2

    def draw(dims, N=None, r=None, fuzz=None, rng=None):
        return np.zeros((N, dims))

    proposal._draw_latent_prior = draw

    with patch(
        "nessai.proposal.flowproposal.flowproposal.partial",
        side_effect=partial,
    ) as mock_partial:
        FlowProposal.prep_latent_prior(proposal)

    mock_partial.assert_called_once_with(
        draw, dims=2, r=3.0, fuzz=1.2, rng=proposal.rng
    )

    assert proposal._draw_func(N=10).shape == (10, 2)


def test_prep_latent_prior_flow(proposal):
    proposal.latent_prior = "flow"
    proposal.flow = MagicMock()
    proposal.flow.sample_latent_distribution = MagicMock()
    FlowProposal.prep_latent_prior(proposal)
    proposal._draw_func(10)
    proposal.flow.sample_latent_distribution.assert_called_once_with(10)


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
def test_populate_accumulate_weights(
    proposal,
    check_acceptance,
    indices,
    r,
    min_radius,
    max_radius,
    wait,
    rng,
):
    """Test the main populate method"""
    n_dims = 2
    poolsize = 10
    drawsize = 5
    names = ["x", "y"]
    worst_point = np.array(
        [[1, 2, 3]], dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")]
    )
    worst_z = rng.standard_normal((1, n_dims))
    z = [
        rng.standard_normal((drawsize, n_dims)),
        rng.standard_normal((drawsize, n_dims)),
        rng.standard_normal((drawsize, n_dims)),
    ]
    x = [
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
    ]
    log_q = [
        np.log(rng.random(drawsize)),
        np.log(rng.random(drawsize)),
        np.log(rng.random(drawsize)),
    ]
    log_w = [
        np.log(np.concatenate([np.ones(drawsize - 1), np.zeros(1)])),
        np.log(np.concatenate([np.ones(drawsize - 1), np.zeros(1)])),
        np.log(np.concatenate([np.ones(drawsize - 1), np.zeros(1)])),
    ]
    # Control rejection sampling using log_w
    rand_u = 0.5 * np.ones(3 * drawsize)

    log_l = rng.random(poolsize)
    log_p = rng.random(poolsize)

    r_flow = 1.0

    if r is None:
        r_out = r_flow
        if min_radius is not None:
            r_out = max(r_out, min_radius)
        if max_radius is not None:
            r_out = min(r_out, max_radius)
    else:
        r_out = r

    proposal.population_time = datetime.timedelta()
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
    proposal.population_dtype = get_dtype(names)
    proposal.truncate_log_q = False
    proposal.use_x_prime_prior = False
    proposal.accumulate_weights = True

    proposal.forward_pass = MagicMock(return_value=(worst_z, np.nan))
    proposal.backward_pass = MagicMock(side_effect=zip(x, log_q))
    proposal.radius = MagicMock(return_value=r_flow)
    proposal.get_alt_distribution = MagicMock(return_value=None)
    proposal.prep_latent_prior = MagicMock()
    proposal.draw_latent_prior = MagicMock(side_effect=z)
    proposal.compute_weights = MagicMock(side_effect=log_w)
    proposal.compute_acceptance = MagicMock(return_value=0.8)
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=log_l
    )
    proposal.rng = MagicMock()
    proposal.rng.random = MagicMock(return_value=rand_u)
    proposal.rng.permutation = rng.permutation

    def convert_to_samples(samples, plot):
        samples["logP"] = log_p
        # wait for windows
        wait()
        return samples

    proposal.plot_pool = MagicMock()
    proposal.convert_to_samples = MagicMock(side_effect=convert_to_samples)

    x_empty = np.empty(0, dtype=proposal.population_dtype)
    with (
        patch(
            "nessai.proposal.flowproposal.flowproposal.empty_structured_array",
            return_value=x_empty,
        ) as mock_empty,
    ):
        FlowProposal.populate(
            proposal, worst_point, n_samples=poolsize, plot=True, r=r
        )

    mock_empty.assert_called_once_with(
        0,
        dtype=proposal.population_dtype,
    )
    proposal.rng.random.assert_called_once_with(3 * drawsize)

    if r is None:
        proposal.forward_pass.assert_called_once_with(
            worst_point,
            rescale=True,
            compute_radius=True,
        )
        proposal.radius.assert_called_once_with(worst_z)
    else:
        assert proposal.r is r

    assert proposal.r == r_out

    proposal.prep_latent_prior.assert_called_once()

    draw_calls = 3 * [call(5)]
    proposal.draw_latent_prior.assert_has_calls(draw_calls)

    backwards_calls = [
        call(
            zz,
            rescale=True,
            return_unit_hypercube=proposal.map_to_unit_hypercube,
        )
        for zz in z
    ]
    proposal.backward_pass.assert_has_calls(backwards_calls)

    compute_weights_calls = [call(xx, lq) for xx, lq in zip(x, log_q)]
    proposal.compute_weights.assert_has_calls(compute_weights_calls)

    proposal.plot_pool.assert_called_once()
    proposal.convert_to_samples.assert_called_once()
    assert_structured_arrays_equal(
        proposal.convert_to_samples.call_args[0][0], proposal.x
    )
    assert proposal.convert_to_samples.call_args[1]["plot"] is True

    assert proposal.population_acceptance == (12 / 15)
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

    assert proposal.population_time.total_seconds() > 0.0


@pytest.mark.parametrize("check_acceptance", [False, True])
@pytest.mark.parametrize("indices", [[], [1]])
@pytest.mark.parametrize("r", [None, 1.0])
@pytest.mark.parametrize(
    "max_radius, min_radius",
    [(0.5, None), (None, 1.5), (None, None)],
)
def test_populate_not_accumulate_weights(
    proposal,
    check_acceptance,
    indices,
    r,
    min_radius,
    max_radius,
    wait,
    rng,
):
    """Test the main populate method"""
    n_dims = 2
    poolsize = 10
    drawsize = 5
    names = ["x", "y"]
    worst_point = np.array(
        [[1, 2, 3]], dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")]
    )
    worst_z = rng.standard_normal((1, n_dims))
    z = [
        rng.standard_normal((drawsize, n_dims)),
        rng.standard_normal((drawsize, n_dims)),
        rng.standard_normal((drawsize, n_dims)),
    ]
    x = [
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
    ]
    log_q = [
        np.log(rng.random(drawsize)),
        np.log(rng.random(drawsize)),
        np.log(rng.random(drawsize)),
    ]
    log_w = [
        np.log(np.concatenate([np.ones(drawsize - 1), np.zeros(1)])),
        np.log(np.concatenate([np.ones(drawsize - 1), np.zeros(1)])),
        np.log(np.concatenate([np.ones(drawsize - 1), np.zeros(1)])),
    ]
    # Control rejection sampling using log_w
    rand_u = [
        0.5 * np.ones(drawsize),
        0.5 * np.ones(drawsize),
        0.5 * np.ones(drawsize),
    ]

    log_l = rng.random(poolsize)
    log_p = rng.random(poolsize)

    r_flow = 1.0

    if r is None:
        r_out = r_flow
        if min_radius is not None:
            r_out = max(r_out, min_radius)
        if max_radius is not None:
            r_out = min(r_out, max_radius)
    else:
        r_out = r

    proposal.population_time = datetime.timedelta()
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
    proposal.population_dtype = get_dtype(names)
    proposal.truncate_log_q = False
    proposal.use_x_prime_prior = False
    proposal.accumulate_weights = False

    proposal.forward_pass = MagicMock(return_value=(worst_z, np.nan))
    proposal.backward_pass = MagicMock(side_effect=zip(x, log_q))
    proposal.radius = MagicMock(return_value=r_flow)
    proposal.get_alt_distribution = MagicMock(return_value=None)
    proposal.prep_latent_prior = MagicMock()
    proposal.draw_latent_prior = MagicMock(side_effect=z)
    proposal.compute_weights = MagicMock(side_effect=log_w)
    proposal.compute_acceptance = MagicMock(return_value=0.8)
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=log_l
    )
    proposal.rng = MagicMock()
    proposal.rng.random = MagicMock(side_effect=rand_u)
    proposal.rng.permutation = rng.permutation

    def convert_to_samples(samples, plot):
        samples["logP"] = log_p
        # wait for windows
        wait()
        return samples

    proposal.plot_pool = MagicMock()
    proposal.convert_to_samples = MagicMock(side_effect=convert_to_samples)

    x_empty = np.empty(poolsize, dtype=proposal.population_dtype)
    with (
        patch(
            "nessai.proposal.flowproposal.flowproposal.empty_structured_array",
            return_value=x_empty,
        ) as mock_empty,
    ):
        FlowProposal.populate(
            proposal, worst_point, n_samples=poolsize, plot=True, r=r
        )

    mock_empty.assert_called_once_with(
        poolsize,
        dtype=proposal.population_dtype,
    )
    proposal.rng.random.assert_has_calls(3 * [call(drawsize)])

    if r is None:
        proposal.forward_pass.assert_called_once_with(
            worst_point,
            rescale=True,
            compute_radius=True,
        )
        proposal.radius.assert_called_once_with(worst_z)
    else:
        assert proposal.r is r

    assert proposal.r == r_out

    proposal.prep_latent_prior.assert_called_once()

    draw_calls = 3 * [call(5)]
    proposal.draw_latent_prior.assert_has_calls(draw_calls)

    backwards_calls = [
        call(
            zz,
            rescale=True,
            return_unit_hypercube=proposal.map_to_unit_hypercube,
        )
        for zz in z
    ]
    proposal.backward_pass.assert_has_calls(backwards_calls)

    compute_weights_calls = [call(xx, lq) for xx, lq in zip(x, log_q)]
    proposal.compute_weights.assert_has_calls(compute_weights_calls)

    proposal.plot_pool.assert_called_once()
    proposal.convert_to_samples.assert_called_once()
    assert_structured_arrays_equal(
        proposal.convert_to_samples.call_args[0][0], proposal.x
    )
    assert proposal.convert_to_samples.call_args[1]["plot"] is True

    assert proposal.population_acceptance == (12 / 15)
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

    assert proposal.population_time.total_seconds() > 0.0


def test_populate_not_initialised(proposal):
    """Assert populate fails if the proposal is not initialised"""
    proposal.initialised = False
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.populate(proposal, 1.0)
    assert "Proposal has not been initialised. " in str(excinfo.value)


def test_populate_truncate_log_q(proposal, rng):
    n_dims = 2
    nlive = 8
    poolsize = 8
    drawsize = 5
    names = ["x", "y"]
    r_flow = 2.0
    worst_point = np.array(
        [[1, 2, 3]], dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")]
    )
    z = [
        rng.standard_normal((drawsize, n_dims)),
        rng.standard_normal((drawsize, n_dims)),
        rng.standard_normal((drawsize, n_dims)),
    ]
    x = [
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
        numpy_array_to_live_points(
            rng.standard_normal((drawsize, n_dims)), names
        ),
    ]
    log_q = [
        np.zeros(drawsize),
        np.zeros(drawsize),
        np.zeros(drawsize),
    ]
    # This sample will be discarded because of the logq min check
    for i in range(3):
        log_q[i][-1] = np.nan_to_num(-np.inf)
    log_w = [
        np.log(np.concatenate([np.ones(drawsize - 2), np.zeros(1)])),
        np.log(np.concatenate([np.ones(drawsize - 2), np.zeros(1)])),
        np.log(np.concatenate([np.ones(drawsize - 2), np.zeros(1)])),
    ]
    # Control rejection sampling using log_w
    rand_u = 0.5 * np.ones(3 * (drawsize - 1))

    log_l = rng.random(poolsize)

    proposal.population_time = datetime.timedelta()
    proposal.initialised = True
    proposal.dims = n_dims
    proposal.poolsize = poolsize
    proposal.drawsize = drawsize
    proposal.fuzz = 1.0
    proposal.indices = None
    proposal.acceptance = [0.7]
    proposal.keep_samples = False
    proposal.fixed_radius = 2.0
    proposal.compute_radius_with_all = False
    proposal.check_acceptance = False
    proposal._plot_pool = False
    proposal.use_x_prime_prior = False
    proposal.populated_count = 1
    proposal.population_dtype = get_dtype(names)
    proposal.truncate_log_q = True
    proposal.training_data = numpy_array_to_live_points(
        np.random.randn(nlive, n_dims),
        names=names,
    )
    proposal.accumulate_weights = True

    log_q_live = np.zeros(nlive)
    log_q_live[-1] = -1.0

    proposal.forward_pass = MagicMock(
        return_value=(nlive * [None], log_q_live)
    )
    proposal.backward_pass = MagicMock(side_effect=zip(x, log_q))
    proposal.compute_weights = MagicMock(side_effect=log_w)
    proposal.radius = MagicMock(return_value=r_flow)
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
    proposal.rng = MagicMock()
    proposal.rng.random = MagicMock(return_value=rand_u)
    proposal.rng.permutation = rng.permutation

    proposal.convert_to_samples = MagicMock(
        side_effect=lambda *args, **kwargs: args[0]
    )

    x_empty = np.empty(0, dtype=proposal.population_dtype)
    with (
        patch(
            "nessai.proposal.flowproposal.flowproposal.empty_structured_array",
            return_value=x_empty,
        ) as mock_empty,
    ):
        FlowProposal.populate(
            proposal, worst_point, n_samples=poolsize, plot=False
        )

    mock_empty.assert_called_once_with(
        0,
        dtype=proposal.population_dtype,
    )
    proposal.rng.random.assert_called_once_with(3 * drawsize - 3)

    assert proposal.population_acceptance == (9 / 15)

    proposal.forward_pass.assert_called_once_with(proposal.training_data)

    backwards_calls = [
        call(
            zz,
            rescale=True,
            return_unit_hypercube=proposal.map_to_unit_hypercube,
        )
        for zz in z
    ]
    proposal.backward_pass.assert_has_calls(backwards_calls)

    compute_weights_calls = [(xx[:-1], lq[:-1]) for xx, lq in zip(x, log_q)]
    for actual_call, expected_call in zip(
        proposal.compute_weights.call_args_list,
        compute_weights_calls,
    ):
        assert_structured_arrays_equal(actual_call[0][0], expected_call[0])
        np.testing.assert_array_equal(actual_call[0][1], expected_call[1])
