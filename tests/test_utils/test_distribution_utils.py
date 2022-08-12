# -*- coding: utf-8 -*-
"""
Test utilities for distributions.
"""
import pytest
from unittest.mock import patch

import torch

from nessai.utils.distributions import (
    get_multivariate_normal,
    get_uniform_distribution,
)


def test_get_uniform_distribution():
    """
    Test function for getting uniform torch distrbution over n dimensions
    """
    with patch("nessai.utils.distributions.BoxUniform") as m:
        get_uniform_distribution(10, 2)
    assert torch.equal(m.call_args_list[0][1]["low"], -2 * torch.ones(10))
    assert torch.equal(m.call_args_list[0][1]["high"], 2 * torch.ones(10))


def test_get_multivariate_normal():
    """Test get multivariate normal"""
    with patch("nessai.utils.distributions.MultivariateNormal") as m:
        get_multivariate_normal(2, var=2)
    assert torch.equal(
        m.call_args_list[0][0][0], torch.zeros(2, dtype=torch.float64)
    )
    assert torch.equal(
        m.call_args_list[0][1]["covariance_matrix"],
        2 * torch.eye(2, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "get_func, args, kwargs",
    [
        (get_multivariate_normal, [], {"var": 2}),
        (get_uniform_distribution, [2], {}),
    ],
)
def test_get_dist_integration(get_func, args, kwargs):
    """Integration test for ge"""
    dist = get_func(3, *args, **kwargs)
    s = dist.sample()
    assert s.get_device() == -1
    dist.log_prob(s)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "get_func, args, kwargs",
    [
        (get_multivariate_normal, [], {"var": 2}),
        (get_uniform_distribution, [2], {}),
    ],
)
def test_get_dist_integration_cuda(get_func, args, kwargs):
    """Integration test for ge"""
    dist = get_func(3, *args, **kwargs, device="cuda")
    s = dist.sample()
    assert s.get_device() != -1
    dist.log_prob(s)
