# -*- coding: utf-8 -*-
"""Specific tests for different included flows."""
import pytest
import torch

from nessai.flows import (
    RealNVP,
    NeuralSplineFlow
)


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(
            net='mlp',
            batch_norm_within_layers=True,
            dropout_probability=0.5
        ),
        dict(linear_transform='permutation'),
        dict(linear_transform='svd'),
        dict(linear_transform='lu'),
        dict(linear_transform=None)
    ]
)
def test_with_realnvp_kwargs(kwargs):
    """Test RealNVP with specific kwargs"""
    flow = RealNVP(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(batch_norm_between_layers=True),
        dict(linear_transform='permutation'),
        dict(linear_transform='svd'),
        dict(linear_transform='lu'),
        dict(linear_transform=None),
        dict(num_bins=10)
    ]
)
def test_with_nsf_kwargs(kwargs):
    """Test NSF with specific kwargs"""
    flow = NeuralSplineFlow(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)
