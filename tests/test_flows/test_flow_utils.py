# -*- coding: utf-8 -*-
"""Test the flow utilities"""
import logging
from nflows.transforms import PointwiseAffineTransform
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, create_autospec, patch

from nessai.flows.utils import (
    MLP,
    configure_model,
    create_linear_transform,
    get_base_distribution,
    get_n_neurons,
    set_affine_parameters,
    silu,
    reset_weights,
    reset_permutations,
)


@pytest.fixture
def config():
    """Minimal config needed for configure_model to work"""
    return dict(
        n_inputs=2,
        n_neurons=4,
        n_blocks=2,
        n_layers=1,
        ftype='realnvp'
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
    mlp.activate_output = False
    x = torch.tensor(1)
    y = torch.tensor(2)
    with patch('nflows.nn.nets.MLP.forward', return_value=y) as parent:
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


def test_configure_model_basic(config):
    """Test configure model with the most basic config."""
    config['kwargs'] = dict(num_bins=2)
    with patch('nessai.flows.realnvp.RealNVP') as mock_flow:
        configure_model(config)

    mock_flow.assert_called_with(
        config['n_inputs'],
        config['n_neurons'],
        config['n_blocks'],
        config['n_layers'],
        num_bins=2
    )


@pytest.mark.parametrize(
    'flow_inputs',
    [
        {'ftype': 'realnvp', 'expected': 'realnvp.RealNVP'},
        {'ftype': 'frealnvp', 'expected': 'realnvp.RealNVP'},
        {'ftype': 'spline', 'expected': 'nsf.NeuralSplineFlow'},
        {'ftype': 'nsf', 'expected': 'nsf.NeuralSplineFlow'},
        {'ftype': 'maf', 'expected': 'maf.MaskedAutoregressiveFlow'}
    ]
)
def test_configure_model_flows(config, flow_inputs):
    """Test the different flows."""
    config['ftype'] = flow_inputs['ftype']
    with patch(f"nessai.flows.{flow_inputs['expected']}") as mock_flow:
        model, _ = configure_model(config)
    mock_flow.assert_called_with(
        config['n_inputs'],
        config['n_neurons'],
        config['n_blocks'],
        config['n_layers'],
    )


def test_configure_model_flow_class(config):
    """Test using a custom class of flow."""
    class TestFlow:
        def __init__(self, n_inputs, n_neurons, n_blocks, n_layers):
            self.n_inputs = n_inputs
            self.n_neurons = n_neurons
            self.n_blocks = n_blocks
            self.n_layers = n_layers

        def to(self, input):
            pass

    config['flow'] = TestFlow
    model, _ = configure_model(config)
    assert isinstance(model, TestFlow)
    assert model.n_inputs == config['n_inputs']
    assert model.n_neurons == config['n_neurons']
    assert model.n_blocks == config['n_blocks']
    assert model.n_layers == config['n_layers']


def test_configure_model_device_cuda(config):
    config['device_tag'] = 'cuda'
    expected_device = torch.device('cuda')
    mock_model = MagicMock()
    with patch('nessai.flows.realnvp.RealNVP',
               return_value=mock_model) as mock_flow:
        model, device = configure_model(config)

    mock_flow.assert_called_with(
        config['n_inputs'],
        config['n_neurons'],
        config['n_blocks'],
        config['n_layers'],
    )

    mock_model.to.assert_called_once_with(expected_device)
    assert model.device == expected_device
    assert device == expected_device


@pytest.mark.parametrize(
    'act',
    [
        {'act': 'relu', 'expected': F.relu},
        {'act': 'tanh', 'expected': F.tanh},
        {'act': 'silu', 'expected': silu},
        {'act': 'swish', 'expected': silu}
    ]
)
def test_configure_model_activation_functions(config, act):
    """Test the different activation functions."""
    config['kwargs'] = dict(activation=act['act'])

    with patch('nessai.flows.realnvp.RealNVP') as mock_flow:
        configure_model(config)

    mock_flow.assert_called_with(
        config['n_inputs'],
        config['n_neurons'],
        config['n_blocks'],
        config['n_layers'],
        activation=act['expected']
    )


def test_configure_model_ftype_error(config):
    """Assert unknown types of flow raise an error."""
    config.pop('ftype')
    with pytest.raises(RuntimeError) as excinfo:
        configure_model(config)
    assert "Must specify either 'flow' or 'ftype'." in str(excinfo.value)


def test_configure_model_input_type_error(config):
    """Assert incorrect type for n_inputs raises an error."""
    config['n_inputs'] = '10'
    with pytest.raises(TypeError) as excinfo:
        configure_model(config)
    assert 'Number of inputs (n_inputs) must be an int' in str(excinfo.value)


def test_configure_model_unknown_activation(config):
    """Assert unknown activation functions raise an error"""
    config['kwargs'] = dict(activation='test')
    with pytest.raises(RuntimeError) as excinfo:
        configure_model(config)
    assert "Unknown activation function: 'test'" in str(excinfo.value)


@pytest.mark.parametrize('linear_transform', ['lu', 'permutation', 'svd'])
def test_create_linear_transform(linear_transform):
    """Test creating a linear transform."""
    lt = create_linear_transform(linear_transform, 2)
    assert lt is not None


def test_create_linear_transform_unknown():
    """Assert an error is raised if an invalid input is given."""
    with pytest.raises(ValueError) as excinfo:
        create_linear_transform('not_a_transform', 2)
    assert 'Unknown linear transform: not_a_transform' in str(excinfo.value)


def test_get_base_distribution_none():
    """Assert that if the distribution is None, None is returned"""
    assert get_base_distribution(2, None) is None


def test_get_base_distribution_class_instance():
    """Assert that if the distribution is a class instance it is returned"""
    dist = MagicMock()
    assert get_base_distribution(2, dist) is dist


def test_get_base_distribution_class():
    """Assert that if the distribution is a class is it correctly called"""
    from nessai.flows.distributions import MultivariateNormal
    dist_cls = MultivariateNormal
    dist = get_base_distribution(2, dist_cls, var=2)
    assert isinstance(dist, MultivariateNormal)
    assert dist._var == 2


@pytest.mark.parametrize(
    "scale, shift",
    [(2.0, 1.0), (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))]
)
def test_set_affine_parameters(scale, shift):
    """Assert the parameters are set correctly"""
    transform = PointwiseAffineTransform()
    set_affine_parameters(transform, scale, shift)
    assert torch.equal(transform._scale, torch.as_tensor(scale))
    assert torch.equal(transform._shift, torch.as_tensor(shift))
    x = torch.randn(5, 2)
    _ = transform.forward(x)
    _ = transform.inverse(x)


@pytest.mark.parametrize(
    "n_neurons, n_inputs, expected",
    [
        (16, 2, 16),
        ('auto', 2, 4),
        ('double', 2, 4),
        (None, 2, 4),
        ('equal', 2, 2),
        ('half', 4, 2),
        (None, None, 8),
    ]
)
def test_get_n_neurons(n_neurons, n_inputs, expected):
    """Assert the correct values are returned"""
    out = get_n_neurons(
        n_neurons=n_neurons,
        n_inputs=n_inputs
    )
    assert isinstance(out, int)
    assert out == expected


@pytest.mark.parametrize(
    "n_neurons, n_inputs",
    [
        ('auto', None),
        ('half', None),
        ('equal', None),
        ('double', None),
        ('invalid', 4)
    ]
)
def test_get_n_neurons_value_error(n_neurons, n_inputs):
    """Assert a ValueError is raised if invalid inputs are given"""
    with pytest.raises(ValueError) as excinfo:
        get_n_neurons(
            n_neurons=n_neurons,
            n_inputs=n_inputs
        )
    assert 'Could not get number of neurons' in str(excinfo.value)
