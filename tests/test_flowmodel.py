
# -*- coding: utf-8 -*-
"""
Test the FlowModel object.
"""
import numpy as np
import pytest
import torch
from unittest.mock import create_autospec, MagicMock, patch

from nessai.flowmodel import update_config, FlowModel


@pytest.fixture()
def data_dim():
    return 2


@pytest.fixture()
def model():
    return create_autospec(FlowModel)


@pytest.fixture(scope='function')
def flow_model(flow_config, data_dim, tmpdir):
    flow_config['model_config']['n_inputs'] = data_dim
    output = str(tmpdir.mkdir('flowmodel'))
    return FlowModel(flow_config, output=output)


def test_update_config_none():
    """Test update config for no input"""
    config = update_config(None)
    assert 'model_config' in config


def test_update_config_invalid_type():
    """Test update config when an invald argument is specified"""
    with pytest.raises(TypeError) as excinfo:
        update_config(False)
    assert 'Must pass a dictionary' in str(excinfo.value)


def test_init_no_config(tmpdir):
    """Test the init method with no config specified"""
    output = str(tmpdir.mkdir('no_config'))
    default_config = update_config(None)
    fm = FlowModel(output=output)

    assert fm.model_config == default_config['model_config']


def test_init_config_class(tmpdir):
    """Test the init and save methods when specifying `flow` as a class"""
    from nessai.flows import FlexibleRealNVP
    output = str(tmpdir.mkdir('no_config'))
    config = dict(model_config=dict(flow=FlexibleRealNVP))
    fm = FlowModel(config=config, output=output)

    assert fm.model_config['flow'].__name__ == 'FlexibleRealNVP'


def test_initialise(model):
    """Test the intialise method"""
    model.get_optimiser = MagicMock()
    model.model_config = dict(n_neurons=2)
    model.inference_device = None
    model.optimiser = 'adam'
    model.optimiser_kwargs = {'weights': 0.1}
    with patch('nessai.flowmodel.setup_model',
               return_value=('model', 'cpu')) as mock:
        FlowModel.initialise(model)
    mock.assert_called_once_with(model.model_config)
    model.get_optimiser.assert_called_once_with('adam', weights=0.1)
    assert model.inference_device == torch.device('cpu')


@pytest.mark.parametrize('optimiser', ['Adam', 'AdamW', 'SGD'])
def test_get_optimiser(model, optimiser):
    """Test to make sure the three supported optimisers work"""
    model.lr = 0.1
    model.model = MagicMock()
    with patch(f'torch.optim.{optimiser}') as mock:
        FlowModel.get_optimiser(model, optimiser)
    mock.assert_called_once()


@pytest.mark.parametrize('val_size, batch_size', [(0.1, 100),
                                                  (0.5, 'all')])
def test_prep_data(flow_model, data_dim, val_size, batch_size):
    """
    Test the data prep, make sure batch sizes and validation size
    produce the correct result.
    """
    n = 100
    x = np.random.randn(n, data_dim)

    train, val = flow_model.prep_data(x, val_size, batch_size)
    if batch_size == 'all':
        batch_size = int(n * (1 - val_size))

    assert flow_model.batch_size == batch_size
    assert train.shape[0] + val.shape[0] == n


@pytest.mark.parametrize('batch_size', [None, '10', True, False])
def test_incorrect_batch_size_type(flow_model, data_dim, batch_size):
    """Ensure the non-interger batch sizes do not work"""
    n = 1000
    x = np.random.randn(n, data_dim)
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.prep_data(x, 0.1, batch_size)
    assert 'Unknown batch size' in str(excinfo.value)


def test_training(flow_model, data_dim):
    """Test class until training"""
    x = np.random.randn(1000, data_dim)
    flow_model.train(x)
    assert flow_model.weights_file is not None


@pytest.mark.parametrize('key, value', [('anneling', True),
                                        ('noise_scale', 0.1),
                                        ('noise_scale', 'adaptive'),
                                        ('max_epochs', 51)])
def test_training_additional_config_args(flow_config, data_dim, tmpdir,
                                         key, value):
    """
    Test training with different config args
    """
    flow_config['model_config']['n_inputs'] = data_dim
    flow_config[key] = value

    output = str(tmpdir.mkdir('flowmodel'))
    flow_model = FlowModel(flow_config, output=output)

    assert getattr(flow_model, key) == value

    x = np.random.randn(100, data_dim)
    flow_model.train(x)


def test_early_optimiser_init(flow_model):
    """Ensure calling the opitmiser before the model raises an error"""
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.get_optimiser()
    assert 'Cannot initialise optimiser' in str(excinfo.value)


def test_sample_and_log_prob_not_initialised(flow_model, data_dim):
    """
    Ensure user cannot call the method before the model initialise.
    """
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.sample_and_log_prob()
    assert 'Model is not initialised' in str(excinfo.value)


@pytest.mark.parametrize('N', [1, 100, 1000])
def test_sample_and_log_prob_no_latent(flow_model, data_dim, N):
    """
    Test the basic use of sample and log prob and ensure correct output
    shapes.
    """
    flow_model.initialise()
    x, log_prob = flow_model.sample_and_log_prob(N=N)
    assert x.shape == (N, data_dim)
    assert log_prob.size == N
