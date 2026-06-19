"""
Test functions related to training and using the flow.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nessai.livepoint import get_dtype
from nessai.proposal import FlowProposal


@pytest.mark.parametrize("log_p", [np.ones(2), np.array([-1, np.inf])])
@pytest.mark.parametrize("discard_nans", [False, True])
@pytest.mark.parametrize("return_z", [False, True])
def test_backward_pass(
    proposal,
    model,
    map_to_unit_hypercube,
    log_p,
    discard_nans,
    return_z,
):
    """Test the forward pass method"""
    n = 2
    if discard_nans:
        acc = int(np.isfinite(log_p).sum())
    else:
        acc = len(log_p)
    x = np.random.randn(n, model.dims)
    z = np.random.randn(n, model.dims)

    def inverse_rescale(a, return_unit_hypercube):
        return a, np.zeros(a.size)

    proposal.inverse_rescale = MagicMock(side_effect=inverse_rescale)
    proposal.prime_parameters = model.names
    proposal._prime_parameters_internal = proposal.prime_parameters
    proposal.x_prime_internal_dtype = get_dtype(
        proposal._prime_parameters_internal
    )
    proposal.check_prior_bounds = MagicMock(
        side_effect=lambda a, b, c: (a, b, c)
    )
    proposal.flow = MagicMock()
    proposal.flow.inverse = MagicMock(return_value=[x, np.zeros(n)])
    proposal.latent_temperature = None
    proposal.compute_latent_log_prob = MagicMock(return_value=log_p)

    out = FlowProposal.backward_pass(
        proposal,
        z,
        discard_nans=discard_nans,
        return_unit_hypercube=map_to_unit_hypercube,
        return_z=return_z,
    )

    x_out = out[0]
    log_p_out = out[1]

    assert len(x_out) == acc
    assert len(log_p_out) == acc
    if return_z:
        assert len(out) == 3
        assert out[2].shape == (acc, model.dims)
    else:
        assert len(out) == 2
    proposal.inverse_rescale.assert_called_once()
    assert (
        proposal.inverse_rescale.call_args.kwargs["return_unit_hypercube"]
        is map_to_unit_hypercube
    )
    proposal.flow.inverse.assert_called_once_with(z)
    proposal.compute_latent_log_prob.assert_called_once_with(z, None)


@pytest.mark.parametrize("return_z", [False, True])
def test_backwards_pass_assertion_error(proposal, caplog, return_z):
    proposal.flow = MagicMock()
    proposal.latent_temperature = None

    def func(*args, **kwargs):
        raise AssertionError("Domain")

    proposal.flow.inverse = MagicMock(side_effect=func)
    out = FlowProposal.backward_pass(
        proposal, np.random.randn(10, 2), return_z=return_z
    )
    assert all([len(a) == 0 for a in out])
    if return_z:
        assert len(out) == 3
    else:
        assert len(out) == 2
    assert "Domain" in caplog.text


def test_backward_pass_with_internal_prime_parameters(
    proposal, model, map_to_unit_hypercube
):
    """Assert hidden prime parameters are retained for inverse rescaling."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    log_p = np.ones(2)
    z = np.random.randn(2, model.dims)

    def inverse_rescale(a, return_unit_hypercube):
        assert a.dtype.names[:3] == ("x_prime", "y_01", "y_prime")
        np.testing.assert_allclose(a["x_prime"], x[:, 0])
        np.testing.assert_allclose(a["y_prime"], x[:, 1])
        return a, np.zeros(a.size)

    proposal.inverse_rescale = MagicMock(side_effect=inverse_rescale)
    proposal.prime_parameters = ["x_prime", "y_prime"]
    proposal._prime_parameters_internal = ["x_prime", "y_01", "y_prime"]
    proposal.x_prime_internal_dtype = get_dtype(
        proposal._prime_parameters_internal
    )
    proposal.check_prior_bounds = MagicMock(
        side_effect=lambda a, b, c: (a, b, c)
    )
    proposal.flow = MagicMock()
    proposal.flow.inverse = MagicMock(return_value=[x, np.zeros(2)])
    proposal.latent_temperature = None
    proposal.compute_latent_log_prob = MagicMock(return_value=log_p)

    out = FlowProposal.backward_pass(
        proposal,
        z,
        return_unit_hypercube=map_to_unit_hypercube,
    )

    assert len(out[0]) == len(log_p)
    proposal.inverse_rescale.assert_called_once()


def test_backward_pass_with_1d_flow_output(
    proposal, model, map_to_unit_hypercube
):
    """Assert 1D flow outputs are promoted to a single-sample batch."""
    x = np.array([1.0, 2.0])
    log_p = np.array([0.0])
    z = np.random.randn(1, model.dims)

    def inverse_rescale(a, return_unit_hypercube):
        assert a.size == 1
        np.testing.assert_allclose(a["x"], [1.0])
        np.testing.assert_allclose(a["y"], [2.0])
        return a, np.zeros(a.size)

    proposal.inverse_rescale = MagicMock(side_effect=inverse_rescale)
    proposal.prime_parameters = model.names
    proposal._prime_parameters_internal = proposal.prime_parameters
    proposal.x_prime_internal_dtype = get_dtype(
        proposal._prime_parameters_internal
    )
    proposal.check_prior_bounds = MagicMock(
        side_effect=lambda a, b, c: (a, b, c)
    )
    proposal.flow = MagicMock()
    proposal.flow.inverse = MagicMock(return_value=[x, np.zeros(1)])
    proposal.latent_temperature = None
    proposal.compute_latent_log_prob = MagicMock(return_value=log_p)

    out = FlowProposal.backward_pass(
        proposal,
        z,
        discard_nans=False,
        return_unit_hypercube=map_to_unit_hypercube,
    )

    assert len(out[0]) == 1
    assert len(out[1]) == 1
    proposal.inverse_rescale.assert_called_once()


def test_sample_latent_distribution_with_temperature(proposal):
    proposal.latent_temperature = 4.0
    proposal.flow = MagicMock()
    proposal.flow.sample_latent_distribution = MagicMock(
        return_value=np.array([[1.0, -2.0]])
    )
    out = proposal.sample_latent_distribution(1)
    proposal.flow.sample_latent_distribution.assert_called_once_with(1)
    np.testing.assert_array_equal(out, np.array([[2.0, -4.0]]))
