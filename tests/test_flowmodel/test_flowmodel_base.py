# -*- coding: utf-8 -*-
"""
Test the FlowModel object.
"""
import json
import numpy as np
import os
import pickle
import pytest
import torch
from unittest.mock import create_autospec, MagicMock, patch

from nessai.flowmodel import FlowModel
from nessai.flowmodel import config
from nessai.flows.realnvp import RealNVP


@pytest.fixture()
def data_dim():
    return 2


@pytest.fixture()
def model():
    return create_autospec(FlowModel)


@pytest.fixture(scope="function")
def flow_model(flow_config, data_dim, tmpdir):
    flow_config["model_config"]["n_inputs"] = data_dim
    output = str(tmpdir.mkdir("flowmodel"))
    return FlowModel(flow_config, output=output)


def test_init_no_config(tmpdir):
    """Test the init method with no config specified"""
    output = str(tmpdir.mkdir("no_config"))
    default_config = config.DEFAULT_FLOW_CONFIG.copy()
    default_config["model_config"] = config.DEFAULT_MODEL_CONFIG.copy()
    fm = FlowModel(output=output)

    assert fm.model_config == default_config["model_config"]


def test_init_no_output(model, tmpdir):
    """Assert the current working directory is used by default"""
    output = str(tmpdir.mkdir("default_output"))
    with patch("os.getcwd", return_value=output) as mock:
        FlowModel.__init__(model, output=None)
    mock.assert_called_once()
    assert model.output is output


def test_init_config_class(tmpdir):
    """Test the init and save methods when specifying `flow` as a class"""

    output = str(tmpdir.mkdir("no_config"))
    config = dict(model_config=dict(flow=RealNVP))
    fm = FlowModel(config=config, output=output)

    assert fm.model_config["flow"].__name__ == "RealNVP"


def test_save_input(model, tmp_path):
    """Test the save input function"""
    output = tmp_path / "test"
    output.mkdir()
    model.output = output

    config = dict(
        patience=10,
        x=np.array([1, 2]),
        model_config=dict(
            n_neurons=10,
            mask=np.array([1, 0]),
            flow=RealNVP,
        ),
    )

    FlowModel.save_input(model, config, output_file=None)

    file_path = os.path.join(output, "flow_config.json")
    assert os.path.exists(file_path)
    with open(file_path, "r") as fp:
        d = json.load(fp)
    assert d["x"] == "[1,2]"
    assert d["model_config"]["mask"] == "[1,0]"


def test_initialise(model):
    """Test the initialise method"""
    model.get_optimiser = MagicMock()
    model.model_config = dict(n_neurons=2)
    model.inference_device = None
    model.optimiser = "adam"
    model.optimiser_kwargs = {"weights": 0.1}
    with patch(
        "nessai.flowmodel.base.configure_model", return_value=("model", "cpu")
    ) as mock:
        FlowModel.initialise(model)
    mock.assert_called_once_with(model.model_config)
    model.get_optimiser.assert_called_once_with("adam", weights=0.1)
    assert model.inference_device == torch.device("cpu")


@pytest.mark.parametrize("optimiser", ["Adam", "AdamW", "SGD"])
def test_get_optimiser(model, optimiser):
    """Test to make sure the three supported optimisers work"""
    model.lr = 0.1
    model.model = MagicMock()
    with patch(f"torch.optim.{optimiser}") as mock:
        FlowModel.get_optimiser(model, optimiser)
    mock.assert_called_once()


@pytest.mark.parametrize(
    "data_size, batch_size",
    [(4010, 1000), (106, 21), (1000, 1000), (2000, 1000)],
)
def test_check_batch_size(data_size, batch_size):
    """"""
    x = np.arange(data_size)
    out = FlowModel.check_batch_size(x, batch_size)
    assert out >= int(0.1 * batch_size)
    if not data_size % batch_size:
        assert out == batch_size


def test_check_batch_size_min_size():
    """Make sure the minimum batch size is respected so long as the final batch
    has size > 1.

    In this case the minimum valid size is 8 but that results in the final
    batch having a size of 1, so it should be 7 instead.
    """
    x = np.arange(25)
    out = FlowModel.check_batch_size(x, 10, min_fraction=0.8)
    assert out == 7


def test_check_batch_size_2():
    """Assert an error is raised if the batch size reaches less than two"""
    x = np.arange(3)
    with pytest.raises(RuntimeError) as excinfo:
        FlowModel.check_batch_size(x, 2, min_fraction=1.0)
    assert "Could not find a valid batch size" in str(excinfo.value)


def test_check_batch_size_1():
    """Assert an error is raised if the batch size is 1"""
    x = np.arange(2)
    with pytest.raises(ValueError) as excinfo:
        FlowModel.check_batch_size(x, 1, min_fraction=1.0)
    assert "Cannot use a batch size of 1" in str(excinfo.value)


@pytest.mark.parametrize("val_size, batch_size", [(0.1, 100), (0.5, "all")])
def test_prep_data(flow_model, data_dim, val_size, batch_size):
    """
    Test the data prep, make sure batch sizes and validation size
    produce the correct result.
    """
    n = 100
    x = np.random.randn(n, data_dim)

    train, val, batch_size_out = flow_model.prep_data(x, val_size, batch_size)
    if batch_size == "all":
        batch_size = int(n * (1 - val_size))

    assert flow_model._batch_size == batch_size
    assert len(train) + len(val) == n


@pytest.mark.parametrize(
    "val_size, batch_size", [(0.1, 100), (0.5, "all"), (0.1, None)]
)
def test_prep_data_dataloader(flow_model, data_dim, val_size, batch_size):
    """
    Test the data prep, make sure batch sizes and validation size
    produce the correct result.
    """
    n = 100
    x = np.random.randn(n, data_dim)

    train, val, batch_size_out = flow_model.prep_data(
        x, val_size, batch_size, use_dataloader=True
    )
    train_batch = next(iter(train))[0]
    val_batch = next(iter(val))[0]
    if batch_size == "all" or batch_size is None:
        batch_size = int(n * (1 - val_size))

    assert train.batch_size == batch_size
    assert list(val_batch.shape) == [int(val_size * n), data_dim]
    assert len(train) * train_batch.shape[0] + val_batch.shape[0] == n


@pytest.mark.parametrize("batch_size", ["10", True, False])
def test_incorrect_batch_size_type(flow_model, data_dim, batch_size):
    """Ensure the non-interger batch sizes do not work"""
    n = 1000
    x = np.random.randn(n, data_dim)
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.prep_data(x, 0.1, batch_size)
    assert "Unknown batch size" in str(excinfo.value)


@pytest.mark.parametrize(
    "x", [np.array([np.inf]), np.array([-np.inf]), np.array([np.nan])]
)
def test_prep_data_non_finite_values(model, x):
    """Assert an error is raised if the samples contain non-finite values"""
    model.initialised = True
    with pytest.raises(
        ValueError, match=r"Cannot train with non-finite samples!"
    ):
        FlowModel.prep_data(model, x, 0.1, 10)


@pytest.mark.parametrize(
    "w", [np.array([np.inf]), np.array([-np.inf]), np.array([np.nan])]
)
def test_prep_data_non_finite_weights(model, w):
    """Assert an error is raised if the weights contain non-finite values"""
    model.initialised = True
    with pytest.raises(ValueError, match=r"Weights contain non-finite values"):
        FlowModel.prep_data(model, np.random.rand(100), 0.1, 50, weights=w)


@pytest.mark.parametrize("dataloader", [False, True])
def test_training(flow_model, data_dim, dataloader):
    """Test class until training"""
    x = np.random.randn(200, data_dim)
    flow_model.use_dataloader = dataloader
    flow_model.train(x)
    assert flow_model.weights_file is not None


def test_training_with_weights(flow_model, data_dim):
    """Test training with weights"""
    x = np.random.randn(200, data_dim)
    weights = np.random.rand(200)
    flow_model.train(x, weights=weights)
    assert flow_model.weights_file is not None


@pytest.mark.parametrize(
    "x", [np.array([np.inf]), np.array([-np.inf]), np.array([np.nan])]
)
def test_training_non_finite_samples(model, x):
    """Assert an error is raised if the samples contain non-finite values"""
    model.initialised = True
    with pytest.raises(ValueError, match=r"Training data is not finite"):
        FlowModel.train(model, x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"annealing": True},
        {"noise_type": "constant", "noise_scale": 0.1},
        {"noise_type": "adaptive", "noise_scale": 0.1},
        {"max_epochs": 51},
        {"val_size": 0.0},
        {"val_size": None},
    ],
)
def test_training_additional_config_args(
    flow_config,
    data_dim,
    tmpdir,
    kwargs,
):
    """
    Test training with different config args
    """
    flow_config["model_config"]["n_inputs"] = data_dim
    for key, value in kwargs.items():
        flow_config[key] = value

    output = str(tmpdir.mkdir("flowmodel"))
    flow_model = FlowModel(flow_config, output=output)

    assert getattr(flow_model, key) == value

    x = np.random.randn(100, data_dim)
    flow_model.train(x)


def test_early_optimiser_init(flow_model):
    """Ensure calling the opitmiser before the model raises an error"""
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.get_optimiser()
    assert "Cannot initialise optimiser" in str(excinfo.value)


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("perms", [False, True])
def test_reset_model(model, weights, perms):
    """Test resetting the model"""
    model.model = MagicMock()
    model.apply = MagicMock()
    model.get_optimiser = MagicMock()
    model.optimiser = MagicMock()
    model.optimiser_kwargs = {"lr": 0.01}

    with patch(
        "nessai.flowmodel.base.configure_model",
        return_value=(MagicMock, "cpu"),
    ) as mock:
        FlowModel.reset_model(model, weights=weights, permutations=perms)

    if weights and perms:
        mock.assert_called_once()
    if any([weights, perms]):
        model.get_optimiser.assert_called_once_with(model.optimiser, lr=0.01)
    else:
        model.get_optimiser.assert_not_called()


def test_sample_and_log_prob_not_initialised(flow_model, data_dim):
    """
    Ensure user cannot call the method before the model initialise.
    """
    with pytest.raises(RuntimeError) as excinfo:
        flow_model.sample_and_log_prob()
    assert "Model is not initialised" in str(excinfo.value)


@pytest.mark.parametrize("N", [1, 100])
def test_sample_and_log_prob_no_latent(flow_model, data_dim, N):
    """
    Test the basic use of sample and log prob and ensure correct output
    shapes.
    """
    flow_model.initialise()
    x, log_prob = flow_model.sample_and_log_prob(N=N)
    assert x.shape == (N, data_dim)
    assert log_prob.size == N


@pytest.mark.parametrize("N", [1, 100])
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


def test_sample_log_prob_alt_dist(model):
    """Assert the alternate distribution is used."""
    z = torch.randn(5, 2)
    x = torch.randn(5, 2)
    log_prob = torch.randn(5)
    log_j = torch.randn(5)
    log_prob_expected = log_prob - log_j
    model.model = MagicMock()
    model.model.device = "cpu"
    model.model.eval = MagicMock()
    model.model.base_distribution_log_prob = MagicMock()
    model.model.inverse = MagicMock(return_value=(x, log_j))
    alt_dist = MagicMock()
    alt_dist.log_prob = MagicMock(return_value=log_prob)

    x_out, log_prob_out = FlowModel.sample_and_log_prob(
        model, z=z, alt_dist=alt_dist
    )

    model.model.inverse.assert_called_once_with(z, context=None)
    model.model.base_distribution_log_prob.assert_not_called()
    alt_dist.log_prob.assert_called_once_with(z)
    np.testing.assert_equal(x_out, x.numpy())
    np.testing.assert_equal(log_prob_out, log_prob_expected)


def test_forward_and_log_prob(model):
    """Assert the method from the flow is called"""
    x = np.random.randn(5, 2)
    log_prob = torch.randn(5)
    z = torch.randn(5, 2)
    model.model = MagicMock()
    model.model.device = "cpu"
    model.model.eval = MagicMock()
    model.model.forward_and_log_prob = MagicMock(return_value=(z, log_prob))

    out_z, out_log_prob = FlowModel.forward_and_log_prob(model, x)

    model.model.eval.assert_called_once()
    np.testing.assert_equal(out_z, z.numpy())
    np.testing.assert_equal(out_log_prob, log_prob.numpy())


def test_log_prob(model):
    """Assert the correct method from the flow is called"""
    x = np.random.randn(5, 2)
    log_prob = torch.randn(5)
    model.model = MagicMock()
    model.model.device = "cpu"
    model.model.eval = MagicMock()
    model.model.log_prob = MagicMock(return_value=log_prob)

    out = FlowModel.log_prob(model, x)

    model.model.eval.assert_called_once()
    np.testing.assert_equal(out, log_prob.numpy())


def test_sample(model):
    """Assert the correct method from the flow is called."""
    n = 10
    x = torch.randn(n, 2)
    model.model = MagicMock()
    model.model.sample = MagicMock(return_value=x)

    out = FlowModel.sample(model, n)

    model.model.sample.assert_called_once_with(n)
    np.testing.assert_array_equal(out, x.numpy())


def test_sample_latent_distribution(model):
    """Assert the correct method is called"""
    n = 10
    z = torch.randn(n, 2)
    model.model = MagicMock()
    model.model.sample_latent_distribution = MagicMock(return_value=z)
    out = FlowModel.sample_latent_distribution(model, n)
    model.model.sample_latent_distribution.assert_called_once_with(n)
    np.testing.assert_array_equal(out, z.numpy())


def test_move_to_update_default(model):
    """Ensure the stored device is updated"""
    model.device = "cuda"
    model.model = MagicMock()
    model.model.to = MagicMock()
    FlowModel.move_to(model, "cpu", update_default=True)
    assert model.device == torch.device("cpu")
    model.model.to.assert_called_once()


@patch("torch.save")
def test_save_weights(mock_save, model):
    """Test saving the weights"""
    model.model = MagicMock()
    FlowModel.save_weights(model, "model.pt")
    mock_save.assert_called_once()
    assert model.weights_file == "model.pt"


@patch("torch.save")
@patch("shutil.move")
@patch("os.path.exists", return_value=True)
def test_save_weights_existing(mock_save, mock_move, mock_exists, model):
    """Assert the file is move to a file with the correct name."""
    model.model = MagicMock()
    FlowModel.save_weights(model, "model.pt")
    mock_save.assert_called_once()
    mock_move.assert_called_once_with("model.pt", "model.pt.old")
    assert model.weights_file == "model.pt"


@pytest.mark.parametrize("initialised", [False, True])
def test_load_weights(model, initialised):
    """Assert the correct functions are called with the correct inputs"""
    weights_file = "test.pt"
    model.initialised = initialised
    model.initialise = MagicMock()
    model.model = MagicMock()
    model.model.load_state_dict = MagicMock()
    model.model.eval = MagicMock()
    d = dict(weight=torch.tensor(1))
    with patch("torch.load", return_value=d) as mock_load:
        FlowModel.load_weights(model, weights_file)
    # Shouldn't initialise twice
    if initialised:
        model.initialise.assert_not_called()
    else:
        model.initialise.assert_called_once()
    mock_load.assert_called_once_with(weights_file)
    model.model.load_state_dict.assert_called_once_with(d)
    model.model.eval.assert_called_once()
    assert model.weights_file == weights_file


def test_reload_weights(model):
    """Assert the correct weights file is used."""
    model.load_weights = MagicMock()
    model.weights_file = "test.pt"
    FlowModel.reload_weights(model, None)
    model.load_weights.assert_called_once_with("test.pt")


def test_get_state(flow_model):
    """Make the object can be pickled"""
    pickle.dumps(flow_model)


@pytest.mark.parametrize("N", [1, 100])
@pytest.mark.integration_test
def test_forward_and_log_prob_integration(flow_model, data_dim, N):
    """Test the basic use of forward and log prob"""
    flow_model.initialise()
    x = np.random.randn(N, data_dim)
    z, log_prob = flow_model.forward_and_log_prob(x)
    assert z.shape == (N, data_dim)
    assert log_prob.size == N


@pytest.mark.integration_test
def test_lu_cache_reset(tmp_path):
    """Assert the LU cache is correctly reset after training.

    Cache is reset when calling .train() so if cache is incorrect then after
    the reset the values in the latent space will not match.
    """
    output = tmp_path / "test"
    output.mkdir()

    config = dict(
        max_epochs=100,
        patience=1000,
        model_config=dict(
            n_inputs=2,
            n_blocks=2,
            kwargs=dict(
                linear_transform="lu",
            ),
        ),
    )

    flow = FlowModel(config=config, output=output)
    data = np.random.randn(100, 2)

    flow.train(data)

    test_data = torch.from_numpy(data).type(torch.get_default_dtype())

    with torch.inference_mode():
        z_out, log_j = flow.model.forward(test_data)
    flow.model.train()
    flow.model.eval()
    with torch.inference_mode():
        z_out_reset, log_j_reset = flow.model.forward(test_data)

    np.testing.assert_array_equal(z_out_reset, z_out)
    np.testing.assert_array_equal(log_j_reset, log_j)


@pytest.mark.integration_test
def test_train_without_validation(tmp_path):
    """Assert training without validation works"""
    output = tmp_path / "test_no_validation"
    output.mkdir()

    config = dict(
        max_epochs=100,
        patience=1000,
        val_size=None,
        model_config=dict(
            n_inputs=2,
            n_blocks=2,
            kwargs=dict(
                linear_transform="lu",
            ),
        ),
    )

    flow = FlowModel(config=config, output=output)
    data = np.random.randn(100, 2)

    history = flow.train(data)

    assert np.isnan(history["val_loss"]).all()
