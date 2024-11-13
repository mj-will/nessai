# -*- coding: utf-8 -*-
"""
Test the FlowModel object.
"""

import json
import os
import pickle
from unittest.mock import MagicMock, Mock, create_autospec, patch

import numpy as np
import pytest
import torch
import torch.utils

from nessai.flowmodel import FlowModel
from nessai.flowmodel import config as default_config
from nessai.flows.realnvp import RealNVP


@pytest.fixture()
def data_dim():
    return 2


@pytest.fixture()
def n_samples():
    return 10


@pytest.fixture(params=[False, True])
def conditional(request, n_samples):
    if request.param:
        return np.random.randn(n_samples, 2)
    else:
        return None


@pytest.fixture()
def model(rng):
    m = create_autospec(FlowModel, rng=rng)
    m.numpy_array_to_tensor = torch.tensor
    m.model = MagicMock()
    return m


@pytest.fixture(scope="function")
def flow_model(flow_config, data_dim, tmpdir, rng):
    flow_config["n_inputs"] = data_dim
    output = str(tmpdir.mkdir("flowmodel"))
    return FlowModel(flow_config, output=output, rng=rng)


def test_init_no_config(tmp_path):
    """Test the init method with no config specified"""
    output = tmp_path / "no_config"
    fm = FlowModel(output=output)
    assert fm.flow_config == default_config.flow.asdict()
    assert fm.training_config == default_config.training.asdict()


def test_init_no_output(model, tmp_path):
    """Assert the current working directory is used by default"""
    output = tmp_path / "default_output"
    with patch("os.getcwd", return_value=output) as mock:
        FlowModel.__init__(model, output=None)
    mock.assert_called_once()
    assert model.output is output


def test_init_config_class(tmp_path):
    """Test the init and save methods when specifying `flow` as a class"""
    output = tmp_path / "config_flow_class"
    flow_config = dict(flow=RealNVP)
    fm = FlowModel(flow_config, output=output)
    assert fm.flow_config["flow"].__name__ == "RealNVP"


def test_save_input(model, tmp_path):
    """Assert the inputs are saved correctly"""
    output = tmp_path / "test"
    output.mkdir()
    model.output = output

    training_config = dict(
        patience=10,
        x=np.array([1, 2]),
    )
    flow_config = dict(
        n_neurons=10,
        mask=np.array([1, 0]),
        flow=RealNVP,
    )

    FlowModel.setup_from_input_dict(model, flow_config, training_config)

    file_path = os.path.join(output, "flow_config.json")
    assert os.path.exists(file_path)
    with open(file_path, "r") as fp:
        d = json.load(fp)
    assert d["mask"] == [1, 0]

    file_path = os.path.join(output, "training_config.json")
    assert os.path.exists(file_path)
    with open(file_path, "r") as fp:
        d = json.load(fp)
    assert d["x"] == [1, 2]


def test_initialise(model):
    """Test the initialise method"""
    model.get_optimiser = MagicMock()
    model.flow_config = dict(n_neurons=2)
    model.inference_device = None
    model.optimiser = "adam"
    model.training_config = dict(optimiser_kwargs={"weights": 0.1})
    mock_flow = MagicMock()
    with patch(
        "nessai.flowmodel.base.configure_model", return_value=mock_flow
    ) as mock:
        FlowModel.initialise(model)
    mock.assert_called_once_with(model.flow_config)
    model.get_optimiser.assert_called_once()
    assert model.inference_device == torch.device("cpu")


@pytest.mark.parametrize("optimiser", ["Adam", "AdamW", "SGD", None])
def test_get_optimiser(model, optimiser):
    """Test to make sure the three supported optimisers work"""
    model.training_config = dict(lr=0.1)
    model.optimiser = "adam"
    model.optimiser_kwargs = dict(beta=0.9)
    model.model = MagicMock()

    optimiser_class = optimiser if optimiser is not None else "Adam"

    with patch(f"torch.optim.{optimiser_class}") as mock:
        FlowModel.get_optimiser(model, optimiser=optimiser, test=True)
    mock.assert_called_once()
    assert mock.call_args.kwargs["lr"] == 0.1
    assert mock.call_args.kwargs["beta"] == 0.9
    assert mock.call_args.kwargs["test"] is True


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


def test_prep_data_conditional(data_dim, rng):
    n = 200
    batch_size = 100
    x = np.random.randn(n, data_dim)
    c = np.random.randn(n, 1)
    fm = create_autospec(FlowModel)
    fm.initialised = True
    fm.check_batch_size = MagicMock(return_value=batch_size)
    fm.rng = rng
    train_loader, val_loader, bs = FlowModel.prep_data(
        fm, x, 0.1, batch_size, conditional=c
    )
    assert bs == batch_size
    assert fm._batch_size == batch_size
    batch = next(iter(train_loader))
    assert len(batch) == 2
    assert batch[0].shape == (batch_size, data_dim)
    assert batch[1].shape == (batch_size, 1)
    batch = next(iter(val_loader))
    assert len(batch) == 2
    assert batch[0].shape == (20, data_dim)
    assert batch[1].shape == (20, 1)


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


def test_prep_data_weights_and_conditional(model):
    """Assert an error is raised if weights and conditional are specified"""
    model.initialised = True
    with pytest.raises(
        RuntimeError, match=r"weights and conditional inputs not supported"
    ):
        FlowModel.prep_data(
            model,
            np.random.rand(100),
            0.1,
            50,
            weights=np.random.rand(100),
            conditional=np.random.rand(100),
        )


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


def test_training_with_conditional(data_dim, tmp_path):
    """Test training with conditional inputs"""
    output = tmp_path / "test_train_conditional"
    fm = create_autospec(FlowModel)
    fm.initialised = True
    fm.device = "cpu"
    fm.inference_device = None
    fm.training_config = dict(
        use_dataloader=False,
        annealing=False,
        patience=20,
        noise_scale=None,
        batch_size=100,
    )
    fm.prep_data = MagicMock(return_value=("train", "val", None))
    fm._train = MagicMock(return_value=1.0)
    fm._validate = MagicMock(return_value=1.0)
    x = np.random.randn(50, data_dim)
    conditional = np.random.randint(0, 2, size=(50, 1))
    FlowModel.train(
        fm,
        x,
        conditional=conditional,
        max_epochs=1,
        val_size=0.1,
        output=output,
    )
    fm.prep_data.assert_called_once_with(
        x,
        val_size=0.1,
        batch_size=100,
        weights=None,
        conditional=conditional,
        use_dataloader=True,
    )
    fm._train.assert_called_once()
    assert fm._train.call_args_list[0][1]["is_conditional"] is True
    fm._validate.assert_called_once()
    assert fm._validate.call_args_list[0][1]["is_conditional"] is True


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
@pytest.mark.parametrize("test_deprecated", [False, True])
def test_training_additional_config_args(
    data_dim,
    tmpdir,
    kwargs,
    test_deprecated,
):
    """
    Test training with different config args
    """
    flow_config = {}
    if test_deprecated:
        flow_config["model_config"] = {}
        flow_config["model_config"]["n_inputs"] = data_dim
        training_config = None
        for key, value in kwargs.items():
            flow_config[key] = value
    else:
        flow_config["n_inputs"] = data_dim
        training_config = {}
        for key, value in kwargs.items():
            training_config[key] = value

    output = str(tmpdir.mkdir("flowmodel"))
    flow_model = FlowModel(
        flow_config,
        training_config=training_config,
        output=output,
    )
    assert flow_model.training_config[key] == value


def test_train_func_conditional(data_dim):
    n = 100

    model = Mock(spec=["train"])

    def log_prob(x, cond):
        assert len(x) == len(cond)
        return torch.randn(x.shape[0], requires_grad=True)

    model.log_prob = MagicMock(side_effect=log_prob)
    model.parameters = MagicMock(return_value=[MagicMock(), MagicMock()])

    fm = create_autospec(FlowModel)
    fm.model = model
    fm._optimiser = MagicMock(spec=torch.optim.Adam)
    fm.device = "cpu"
    fm.training_config = default_config.training.asdict()

    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(n, data_dim),
            torch.randn(n, 1),
        ),
        batch_size=n,
    )

    FlowModel._train(fm, data, is_dataloader=True, is_conditional=True)

    model.log_prob.assert_called_once()


def test_validate_func_conditional(data_dim):
    n = 100

    model = Mock(spec=["eval"])

    def log_prob(x, cond):
        assert len(x) == len(cond)
        return torch.randn(x.shape[0], requires_grad=True)

    model.log_prob = MagicMock(side_effect=log_prob)
    model.parameters = MagicMock(return_value=[MagicMock(), MagicMock()])

    fm = create_autospec(FlowModel)
    fm.model = model
    fm._optimiser = MagicMock(spec=torch.optim.Adam)
    fm.device = "cpu"
    fm.training_config = default_config.training.asdict()

    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(n, data_dim),
            torch.randn(n, 1),
        ),
        batch_size=n,
    )

    FlowModel._validate(fm, data, is_dataloader=True, is_conditional=True)

    model.log_prob.assert_called_once()


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
    model.flow_config = dict(
        n_inputs=2,
    )
    model.training_config = dict(lr=0.1)
    model.optimiser_kwargs = dict(beta=0.9)

    with patch(
        "nessai.flowmodel.base.configure_model",
        return_value=MagicMock,
    ) as mock:
        FlowModel.reset_model(model, weights=weights, permutations=perms)

    if weights and perms:
        mock.assert_called_once()
    if any([weights, perms]):
        model.get_optimiser.assert_called_once()
    else:
        model.get_optimiser.assert_not_called()


def test_sample_and_log_prob(flow_model, n_samples, conditional):
    """Assert the outputs have the correct shape"""
    if conditional is not None:
        flow_model.flow_config["context_features"] = conditional.shape[-1]
    flow_model.initialise()
    samples, log_prob = flow_model.sample_and_log_prob(
        n_samples, conditional=conditional
    )
    assert len(samples) == n_samples
    assert len(log_prob) == n_samples


def test_sample_and_log_prob_not_initialised(flow_model):
    """
    Ensure user cannot call the method before the model initialise.
    """
    with pytest.raises(RuntimeError, match="Model is not initialised"):
        flow_model.sample_and_log_prob()


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


def test_numpy_array_to_tensor(model, n_samples):
    x = np.random.randn(n_samples, 2).astype("float32")
    model.model.device = "cpu"
    out = FlowModel.numpy_array_to_tensor(model, x)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.get_default_dtype()
    np.testing.assert_equal(x, out.numpy())


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


def test_forward_and_log_prob(model, n_samples, conditional):
    """Assert the method from the flow is called"""
    x = np.random.randn(n_samples, 2)
    log_prob = torch.randn(n_samples)
    z = torch.randn(n_samples, 2)
    model.model = MagicMock()
    model.model.device = "cpu"
    model.model.eval = MagicMock()
    model.model.forward_and_log_prob = MagicMock(return_value=(z, log_prob))

    out_z, out_log_prob = FlowModel.forward_and_log_prob(
        model, x, conditional=conditional
    )

    model.model.eval.assert_called_once()
    model.model.forward_and_log_prob.assert_called_once()
    if conditional is not None:
        assert np.array_equal(
            model.model.forward_and_log_prob.call_args_list[0][1]["context"],
            conditional,
        )
    np.testing.assert_equal(out_z, z.numpy())
    np.testing.assert_equal(out_log_prob, log_prob.numpy())


def test_inverse(model, n_samples, conditional):
    z = np.random.randn(n_samples, 2)
    log_j = torch.randn(n_samples)
    x = torch.randn(n_samples, 2)
    model.model = MagicMock()
    model.model.device = "cpu"
    model.model.eval = MagicMock()
    model.model.inverse = MagicMock(return_value=(x, log_j))

    out_x, out_log_j = FlowModel.inverse(model, z, conditional=conditional)

    model.model.eval.assert_called_once()
    model.model.inverse.assert_called_once()
    if conditional is not None:
        assert np.array_equal(
            model.model.inverse.call_args_list[0][1]["context"],
            conditional,
        )
    np.testing.assert_equal(out_x, x.numpy())
    np.testing.assert_equal(out_log_j, log_j.numpy())


def test_log_prob(model, n_samples, conditional):
    """Assert the correct method from the flow is called"""
    x = np.random.randn(n_samples, 2)
    log_prob = torch.randn(n_samples)
    model.model = MagicMock()
    model.model.device = "cpu"
    model.model.eval = MagicMock()
    model.model.log_prob = MagicMock(return_value=log_prob)

    out = FlowModel.log_prob(model, x, conditional=conditional)

    model.model.eval.assert_called_once()
    if conditional is not None:
        assert np.array_equal(
            model.model.log_prob.call_args_list[0][1]["context"], conditional
        )
    np.testing.assert_equal(out, log_prob.numpy())


def test_sample(model, n_samples, conditional):
    """Assert the correct method from the flow is called."""
    x = torch.randn(n_samples, 2)
    model.model = MagicMock()
    model.model.sample = MagicMock(return_value=x)

    out = FlowModel.sample(model, n_samples, conditional=conditional)

    model.model.sample.assert_called_once()
    assert model.model.sample.call_args_list[0][0][0] == n_samples
    if conditional is not None:
        assert np.array_equal(
            model.model.sample.call_args_list[0][1]["context"], conditional
        )
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
@pytest.mark.parametrize("conditional", [None, True])
@pytest.mark.integration_test
def test_forward_and_log_prob_integration(
    flow_model, data_dim, N, conditional
):
    """Test the basic use of forward and log prob"""
    if conditional:
        conditional = np.random.randn(N, 1)
        flow_model.flow_config["context_features"] = 1
    flow_model.initialise()
    x = np.random.randn(N, data_dim)
    z, log_prob = flow_model.forward_and_log_prob(x, conditional=conditional)
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

    training_config = dict(
        max_epochs=100,
        patience=1000,
    )
    flow_config = dict(
        n_inputs=2,
        n_blocks=2,
        linear_transform="lu",
    )

    flow = FlowModel(
        flow_config=flow_config, training_config=training_config, output=output
    )
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

    training_config = dict(
        max_epochs=100,
        patience=1000,
        val_size=0,
    )
    flow_config = dict(
        n_inputs=2,
        n_blocks=2,
        linear_transform="lu",
    )

    flow = FlowModel(
        flow_config=flow_config, training_config=training_config, output=output
    )
    data = np.random.randn(100, 2)

    history = flow.train(data)

    assert np.isnan(history["val_loss"]).all()


@pytest.mark.integration_test
def test_train_conditional_integration(tmp_path):
    """Assert training with conditional data works"""
    output = tmp_path / "test_train_conditional"
    output.mkdir()

    training_config = dict(
        max_epochs=10,
    )
    flow_config = dict(
        n_inputs=2,
        n_blocks=2,
        linear_transform="lu",
        context_features=1,
    )

    flow = FlowModel(
        flow_config=flow_config, training_config=training_config, output=output
    )
    data = np.random.randn(100, 2)
    conditional = np.random.randint(2, size=(100, 1))

    _ = flow.train(data, conditional=conditional)
