
# -*- coding: utf-8 -*-
"""
Test the FlowModel object.
"""
import numpy as np
import pickle
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
    """Test update config when an invalid argument is specified"""
    with pytest.raises(TypeError) as excinfo:
        update_config(False)
    assert 'Must pass a dictionary' in str(excinfo.value)


@pytest.mark.parametrize('noise_scale', ['auto', 4])
def test_update_config_invalid_noise_scale(noise_scale):
    """Assert an error is raised if noise_scale is not a float or adapative."""
    config = {'noise_scale': noise_scale}
    with pytest.raises(ValueError) as excinfo:
        update_config(config)
    assert 'noise_scale must be a float or' in str(excinfo.value)


def test_update_config_n_neurons():
    """Assert the n_neurons is set to 2x n_inputs"""
    config = dict(model_config=dict(n_inputs=10))
    config = update_config(config)
    assert config['model_config']['n_neurons'] == 20


def test_init_no_config(tmpdir):
    """Test the init method with no config specified"""
    output = str(tmpdir.mkdir('no_config'))
    default_config = update_config(None)
    fm = FlowModel(output=output)

    assert fm.model_config == default_config['model_config']


def test_init_config_class(tmpdir):
    """Test the init and save methods when specifying `flow` as a class"""
    from nessai.flows import RealNVP
    output = str(tmpdir.mkdir('no_config'))
    config = dict(model_config=dict(flow=RealNVP))
    fm = FlowModel(config=config, output=output)

    assert fm.model_config['flow'].__name__ == 'RealNVP'


def test_initialise(model):
    """Test the initialise method"""
    model.get_optimiser = MagicMock()
    model.model_config = dict(n_neurons=2)
    model.inference_device = None
    model.optimiser = 'adam'
    model.optimiser_kwargs = {'weights': 0.1}
    with patch('nessai.flowmodel.configure_model',
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
    assert len(train) + len(val) == n


@pytest.mark.parametrize('val_size, batch_size', [(0.1, 100),
                                                  (0.5, 'all')])
def test_prep_data_dataloader(flow_model, data_dim, val_size, batch_size):
    """
    Test the data prep, make sure batch sizes and validation size
    produce the correct result.
    """
    n = 100
    x = np.random.randn(n, data_dim)

    train, val = flow_model.prep_data(
        x, val_size, batch_size, use_dataloader=True)
    train_batch = next(iter(train))[0]
    val_batch = next(iter(val))[0]
    if batch_size == 'all':
        batch_size = int(n * (1 - val_size))

    assert train.batch_size == batch_size
    assert list(val_batch.shape) == [int(val_size * n), data_dim]
    assert len(train) * train_batch.shape[0] + val_batch.shape[0] == n


@pytest.mark.parametrize('batch_size', [None, '10', True, False])
def test_incorrect_batch_size_type(flow_model, data_dim, batch_size):
    """Ensure the non-interger batch sizes do not work"""
    n = 1000
    x = np.random.randn(n, data_dim)
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.prep_data(x, 0.1, batch_size)
    assert 'Unknown batch size' in str(excinfo.value)


@pytest.mark.parametrize('dataloader', [False, True])
def test_training(flow_model, data_dim, dataloader):
    """Test class until training"""
    x = np.random.randn(1000, data_dim)
    flow_model.use_dataloader = dataloader
    flow_model.train(x)
    assert flow_model.weights_file is not None


@pytest.mark.parametrize('key, value', [('annealing', True),
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


@pytest.mark.parametrize('weights', [False, True])
@pytest.mark.parametrize('perms', [False, True])
def test_reset_model(flow_model, weights, perms):
    """Test resetting the model"""
    flow_model.initialise()
    flow_model.reset_model(weights=weights, permutations=perms)


def test_sample_and_log_prob_not_initialised(flow_model, data_dim):
    """
    Ensure user cannot call the method before the model initialise.
    """
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.sample_and_log_prob()
    assert 'Model is not initialised' in str(excinfo.value)


@pytest.mark.parametrize('N', [1, 100])
def test_sample_and_log_prob_no_latent(flow_model, data_dim, N):
    """
    Test the basic use of sample and log prob and ensure correct output
    shapes.
    """
    flow_model.initialise()
    x, log_prob = flow_model.sample_and_log_prob(N=N)
    assert x.shape == (N, data_dim)
    assert log_prob.size == N


@pytest.mark.parametrize('N', [1, 100])
def test_sample_and_log_prob_with_latent(flow_model, data_dim, N):
    """
    Test the basic use of sample and log prob when samples from the
    latent space are provided
    """
    flow_model.initialise()
    z = np.random.randn(N, data_dim)
    x, log_prob = flow_model.sample_and_log_prob(z=z)
    assert x.shape == (N, data_dim)
    assert log_prob.size == N


@pytest.mark.parametrize('N', [1, 100])
def test_forward_and_log_prob(flow_model, data_dim, N):
    """Test the basic use of forward and log prob"""
    flow_model.initialise()
    x = np.random.randn(N, data_dim)
    z, log_prob = flow_model.forward_and_log_prob(x)
    assert z.shape == (N, data_dim)
    assert log_prob.size == N


def test_move_to_update_default(model):
    """Ensure the stored device is updated"""
    model.device = 'cuda'
    model.model = MagicMock()
    model.model.to = MagicMock()
    FlowModel.move_to(model, 'cpu', update_default=True)
    assert model.device == torch.device('cpu')
    model.model.to.assert_called_once()


@patch('torch.save')
def test_save_weights(mock_save, model):
    """Test saving the weights"""
    model.model = MagicMock()
    FlowModel.save_weights(model, 'model.pt')
    mock_save.assert_called_once()
    assert model.weights_file == 'model.pt'


def test_get_state(flow_model):
    """Make the object can be pickled"""
    pickle.dumps(flow_model)
