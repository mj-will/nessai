# -*- coding: utf-8 -*-
"""
Tests for Pyro based flow.
"""
import numpy as np
import pytest
import torch

from nessai.flows.base.pyro import PyroFlow


@pytest.mark.requires('pyro')
@pytest.mark.integration_test
def test_flow_integration():
    """Assert the base class work as intended."""
    import pyro.distributions as dist
    import pyro.distributions.transforms as T
    base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
    spline_transform = T.spline_coupling(2, count_bins=16)
    flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])
    flow = PyroFlow(flow_dist)

    x = flow.sample(10)
    z, log_j = flow.forward(x)
    x_out, log_j_inv = flow.inverse(z)

    log_prob = flow.log_prob(x)
    log_prob_alt = flow.base_distribution_log_prob(z) + log_j

    np.testing.assert_array_equal(x_out.detach().cpu(), x.detach().cpu())
    np.testing.assert_array_equal(
        log_j_inv.detach().cpu(), -log_j.detach().cpu()
    )
    np.testing.assert_array_equal(
        log_prob.detach().cpu(), log_prob_alt.detach().cpu()
    )
