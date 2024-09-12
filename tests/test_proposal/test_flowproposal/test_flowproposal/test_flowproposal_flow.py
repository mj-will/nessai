"""
Test functions related to training and using the flow.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

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
    proposal.alt_dist = None
    proposal.check_prior_bounds = MagicMock(
        side_effect=lambda a, b, c: (a, b, c)
    )
    proposal.flow = MagicMock()
    proposal.flow.sample_and_log_prob = MagicMock(return_value=[x, log_p])

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
    proposal.flow.sample_and_log_prob.assert_called_once_with(
        z=z, alt_dist=None
    )


@pytest.mark.parametrize("return_z", [False, True])
def test_backwards_pass_assertion_error(proposal, caplog, return_z):
    proposal.alt_dist = None
    proposal.flow = MagicMock()

    def func(*args, **kwargs):
        raise AssertionError("Domain")

    proposal.flow.sample_and_log_prob = MagicMock(side_effect=func)
    out = FlowProposal.backward_pass(
        proposal, np.random.randn(10, 2), return_z=return_z
    )
    assert all([len(a) == 0 for a in out])
    if return_z:
        assert len(out) == 3
    else:
        assert len(out) == 2
    assert "Domain" in caplog.text
