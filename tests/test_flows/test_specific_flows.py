# -*- coding: utf-8 -*-
"""Specific tests for different included flows."""
import numpy as np
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
        dict(use_volume_preserving=True),
        dict(linear_transform='permutation'),
        dict(linear_transform='svd'),
        dict(linear_transform='lu'),
        dict(linear_transform=None),
        dict(mask=np.array([[1, -1], [-1, 1]])),
        dict(mask=[1, -1]),
    ]
)
def test_with_realnvp_kwargs(kwargs):
    """Test RealNVP with specific kwargs"""
    flow = RealNVP(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)


@pytest.mark.parametrize(
    'kwargs, string',
    [
        (dict(net='res'), 'Unknown nn type: res'),
        (dict(linear_transform='test'), 'Unknown linear transform: test'),
        (dict(mask=[1, 1, -1]), 'Mask does not match number of features'),
        (dict(mask=[[-1, 1], [1, -1], [1, -1]]),
         'Mask does not match number of layers'),
    ]
)
def test_realnvp_value_errors(kwargs, string):
    """Assert incorrect values for some inputs raise an error"""
    with pytest.raises(ValueError) as excinfo:
        RealNVP(2, 2, 2, 2, **kwargs)
    assert string in str(excinfo.value)


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


@pytest.mark.parametrize('FlowClass', [RealNVP, NeuralSplineFlow])
def test_1d_inputs(FlowClass):
    """Assert an error is raised if 1-d inputs are specified."""
    with pytest.raises(ValueError) as excinfo:
        FlowClass(1, 2, 2, 2)
    assert 'requires at least 2 dimensions' in str(excinfo.value)
