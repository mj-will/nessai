# -*- coding: utf-8 -*-
"""Test the flow utilties"""
import logging
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, create_autospec, patch

from nessai.flows.utils import (
    silu,
    reset_weights,
    reset_permutations,
    MLP
)


def test_silu():
    """Test the silu activation"""
    from scipy.special import expit
    x = torch.randn(100)
    y = silu(x)
    expected = x.numpy() * expit(x.numpy())
    np.testing.assert_array_almost_equal(y, expected)


def test_reset_weights_with_reset_parameters():
    """Test the reset weights function for module with ``reset_parameters``"""
    module = MagicMock()
    module.reset_parameters = MagicMock()
    reset_weights(module)
    module.reset_parameters.assert_called_once()


def test_reset_weights_batch_norm():
    """Test the reset weights function for an instance of batch norm"""
    from nflows.transforms.normalization import BatchNorm
    x = torch.randn(20, 2)
    module = BatchNorm(2, eps=0.1)
    module.train()
    module.forward(x)

    reset_weights(module)

    constant = np.log(np.exp(1 - 0.1) - 1)
    assert (module.unconstrained_weight.data == constant).all()
    assert (module.bias.data == 0).all()
    assert (module.running_mean == 0).all()
    assert (module.running_var == 1).all()


def test_reset_weights_other_module(caplog):
    """Test reset weights on a module that cannot be reset."""
    caplog.set_level(logging.WARNING)
    module = object
    reset_weights(module)
    assert 'Could not reset' in caplog.text


def test_weight_reset_permutation():
    """Test to make sure random permutation is reset correctly"""
    from nflows.transforms.permutations import RandomPermutation
    x = torch.arange(10).reshape(1, -1)
    m = RandomPermutation(features=10)
    y_init, _ = m(x)
    p = m._permutation.numpy()
    m.apply(reset_permutations)
    y_reset, _ = m(x)
    assert not (p == m._permutation.numpy()).all()
    assert not (y_init.numpy() == y_reset.numpy()).all()


def test_mlp_forward():
    """Test the MLP implementation of forward"""
    mlp = create_autospec(MLP)
    x = torch.tensor(1)
    y = torch.tensor(2)
    with patch('nflows.nn.nets.MLP.forward',
               return_value=y) as parent:
        out = MLP.forward(mlp, x, context=None)
    parent.assert_called_once_with(x)
    assert out == y


def test_mlp_forward_context():
    """Assert an error is raised in the context is not None"""
    mlp = create_autospec(MLP)
    x = torch.tensor(1)
    with pytest.raises(NotImplementedError) as excinfo:
        MLP.forward(mlp, x, context=x)
    assert 'MLP with conditional inputs is not implemented.' in \
        str(excinfo.value)
