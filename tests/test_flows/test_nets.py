# -*- coding: utf-8 -*-
"""Tests for the nessai.flows.nets submodule"""

from unittest.mock import create_autospec

import torch
from torch import nn
import torch.nn.functional as F
import pytest

from nessai.flows.nets import MLP


@pytest.fixture
def mlp():
    return create_autospec(MLP)


def test_mlp_init_defaults(mlp):
    """Assert the default init returns the correct network"""
    MLP.__init__(mlp, (2,), (1,), [64, 64])
    assert len(mlp._hidden_layers) == 1


@pytest.mark.parametrize("activate_output", [False, True, F.relu])
def test_mlp_init_activate_output(mlp, activate_output):
    """Assert the default init returns the correct network"""
    MLP.__init__(mlp, (2,), (1,), [64, 64], activate_output=activate_output)


def test_init_invalid_hidden(mlp):
    """Assert an error is raised if the hidden sizes list is empty"""
    with pytest.raises(ValueError) as excinfo:
        MLP.__init__(mlp, (2,), (1,), [])
    assert "List of hidden sizes" in str(excinfo.value)


def test_init_invalid_output_act(mlp):
    """Assert an error is raised activate output is the wrong type"""
    with pytest.raises(TypeError) as excinfo:
        MLP.__init__(mlp, (2,), (1,), [10], activate_output="relu")
    assert "must be a boolean or a callable" in str(excinfo.value)


def test_mlp_forward(mlp):
    """Test the MLP implementation of forward"""
    x = torch.rand(4, 2)
    # All of the components of the MLP
    mlp._in_shape = torch.Size((2,))
    mlp._out_shape = torch.Size((1,))
    mlp._input_layer = nn.Linear(2, 10)
    mlp._hidden_layers = [nn.Linear(10, 10)]
    mlp._dropout_layers = [nn.Dropout(0.5)]
    mlp._output_layer = nn.Linear(10, 1)
    mlp._activate_output = F.sigmoid
    mlp._activation = F.relu
    mlp._output_activation = F.sigmoid
    out = MLP.forward(mlp, x, context=None)
    assert out.shape == torch.Size((4, 1))
    assert all(out >= 0.0)
    assert all(out <= 1.0)


def test_mlp_forward_context(mlp):
    """Assert an error is raised in the context is not None"""
    x = torch.tensor(1)
    with pytest.raises(ValueError) as excinfo:
        MLP.forward(mlp, x, context=x)
    assert "MLP with conditional inputs is not implemented." in str(
        excinfo.value
    )


def test_mlp_forward_incorrect_shape(mlp):
    """Assert an error is raised if the input shape is incorrect"""
    x = torch.rand(10, 2)
    mlp._in_shape = torch.Size((3,))
    with pytest.raises(ValueError) as excinfo:
        MLP.forward(mlp, x)
    assert "Expected inputs of shape" in str(excinfo.value)
