"""
Test functions related to training and using the flow.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from nessai.proposal import FlowProposal


@pytest.mark.parametrize("log_p", [np.ones(2), np.array([-1, np.inf])])
@pytest.mark.parametrize("discard_nans", [False, True])
def test_backward_pass(
    proposal, model, log_p, discard_nans, map_to_unit_hypercube
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

    x_out, log_p = FlowProposal.backward_pass(
        proposal,
        z,
        discard_nans=discard_nans,
        return_unit_hypercube=map_to_unit_hypercube,
    )

    assert len(x_out) == acc
    proposal.inverse_rescale.assert_called_once()
    assert (
        proposal.inverse_rescale.call_args.kwargs["return_unit_hypercube"]
        is map_to_unit_hypercube
    )
    proposal.flow.sample_and_log_prob.assert_called_once_with(
        z=z, alt_dist=None
    )
