"""
Test the ImportanceFlowModel.
"""

import os
import pickle
from unittest.mock import MagicMock, create_autospec, patch

from nessai.flowmodel.importance import ImportanceFlowModel as IFM
import numpy as np
import pytest
import torch


@pytest.fixture()
def ifm():
    return create_autospec(IFM)


class DummyFlow(torch.nn.Module):
    pass


def test_init(ifm):
    config = dict(patience=20)
    output = "test"
    with patch("nessai.flowmodel.importance.FlowModel.__init__") as mock_init:
        IFM.__init__(ifm, output=output, config=config)
    mock_init.assert_called_once_with(config=config, output=output)
    assert ifm.weights_files == []
    assert len(ifm.models) == 0


def test_model_property_with_models(ifm):
    """Assert last model is returned"""
    models = [DummyFlow(), DummyFlow(), DummyFlow()]
    ifm.models = models
    assert IFM.model.__get__(ifm) is models[-1]


def test_model_property_without_models(ifm):
    """Assert None is return if no models have been defined"""
    ifm.models = []
    assert IFM.model.__get__(ifm) is None


def test_model_setter(ifm):
    """Assert the model setter appends the model"""
    new_model = DummyFlow()
    models = [DummyFlow(), DummyFlow()]
    ifm.models = models
    IFM.model.__set__(ifm, new_model)
    assert len(ifm.models) == 3
    assert ifm.models[-1] is new_model


def test_model_setter_none(ifm):
    """Assert none is not added to the models"""
    new_model = None
    models = [DummyFlow(), DummyFlow()]
    ifm.models = models
    IFM.model.__set__(ifm, new_model)
    assert len(ifm.models) == 2
    assert None not in ifm.models


def test_n_models(ifm):
    """Assert the number of models is correct"""
    ifm.models = [DummyFlow(), DummyFlow()]
    assert IFM.n_models.__get__(ifm) == 2


@pytest.mark.parametrize("models", [None, torch.nn.ModuleList()])
def test_n_models_no_models(ifm, models):
    """Assert the number of models is 0 if models is empty or None"""
    ifm.models = models
    assert IFM.n_models.__get__(ifm) == 0


def test_initialise(ifm):
    """Assert initialise sets the correct boolean"""
    ifm.initialised = False
    IFM.initialise(ifm)
    assert ifm.initialised is True


def test_reset_optimiser(ifm):
    """Assert the optimiser is reset"""
    optimiser = "adam"
    optimiser_kwargs = dict(decay=0.1)
    _optimiser = object()
    ifm.get_optimiser = MagicMock(return_value=_optimiser)
    ifm.optimiser = optimiser
    ifm.optimiser_kwargs = optimiser_kwargs
    IFM.reset_optimiser(ifm)
    ifm.get_optimiser.assert_called_once_with(
        optimiser=optimiser,
        **optimiser_kwargs,
    )
    assert ifm._optimiser is _optimiser


def test_add_new_flow_reset(ifm):
    """Assert a new flow is created when reset=True."""
    flow = DummyFlow()
    device = torch.device("cpu")

    ifm.models = torch.nn.ModuleList([DummyFlow(), DummyFlow()])
    ifm.model_config = dict(inference_device_tag=None)
    ifm.reset_optimiser = MagicMock()

    with patch(
        "nessai.flowmodel.importance.configure_model",
        return_value=(flow, device),
    ) as mock_configure:
        IFM.add_new_flow(ifm, reset=True)

    assert len(ifm.models) == 3
    assert ifm.models[-1] is flow
    assert ifm.models.training is False

    mock_configure.assert_called_once_with(ifm.model_config)

    ifm.reset_optimiser.assert_called_once()
    assert ifm.device == device
    assert ifm.inference_device == device


def test_add_new_flow_no_reset(ifm):
    """Assert the flow is copied when reset=False"""
    flow = DummyFlow()
    device = torch.device("cpu")

    flow_to_copy = DummyFlow()

    ifm.device = device
    ifm.model = flow_to_copy
    ifm.models = torch.nn.ModuleList([DummyFlow(), flow_to_copy])
    ifm.model_config = dict(inference_device_tag=None)
    ifm.reset_optimiser = MagicMock()

    with patch(
        "copy.deepcopy",
        return_value=flow,
    ) as mock_copy:
        IFM.add_new_flow(ifm, reset=False)

    assert len(ifm.models) == 3
    assert ifm.models[-1] is flow
    assert ifm.models.training is False

    mock_copy.assert_called_once_with(ifm.models[-2])

    ifm.reset_optimiser.assert_called_once()
    assert ifm.device == device
    assert ifm.inference_device == device


def test_add_new_flow_first_flow(ifm):
    """Assert a new flow is created when no flows have been added"""
    flow = DummyFlow()
    device = torch.device("cpu")

    ifm.models = torch.nn.ModuleList()
    ifm.model_config = dict(inference_device_tag=None)
    ifm.reset_optimiser = MagicMock()

    with patch(
        "nessai.flowmodel.importance.configure_model",
        return_value=(flow, device),
    ) as mock_configure:
        IFM.add_new_flow(ifm, reset=False)

    assert len(ifm.models) == 1
    assert ifm.models[-1] is flow
    assert ifm.models.training is False

    mock_configure.assert_called_once_with(ifm.model_config)

    ifm.reset_optimiser.assert_called_once()
    assert ifm.device == device
    assert ifm.inference_device == device


def test_add_new_flow_first_inference_device(ifm):
    """Assert the inference device is set correctly"""
    flow = DummyFlow()
    device = torch.device("cpu")

    ifm.models = torch.nn.ModuleList()
    ifm.model_config = dict(inference_device_tag="cuda")

    dummy_device = object()

    with patch(
        "nessai.flowmodel.importance.configure_model",
        return_value=(flow, device),
    ), patch("torch.device", return_value=dummy_device) as mock_device:
        IFM.add_new_flow(ifm, reset=False)

    mock_device.assert_called_once_with("cuda")
    assert ifm.inference_device is dummy_device


@pytest.mark.parametrize("i", [0, 1, 2, -1])
def test_log_prob_th(ifm, i):
    """Assert correct values are returned"""
    x = np.random.randn(10, 2)
    log_prob = torch.ones(len(x), dtype=torch.float32)

    model = MagicMock(spec=DummyFlow)
    model.log_prob = MagicMock(return_value=log_prob)
    model.device = torch.device("cpu")
    model.training = True
    model.eval = MagicMock()

    models = [DummyFlow(), DummyFlow(), DummyFlow()]
    models[i] = model
    ifm.models = torch.nn.ModuleList(models)
    ifm.models.training = True

    out = IFM.log_prob_ith(ifm, x, i)

    assert out.dtype == np.float64
    model.eval.assert_called_once()
    np.testing.assert_array_equal(out, log_prob.numpy())


@pytest.mark.parametrize("exclude_last", [False, True])
def test_log_prob_all(ifm, exclude_last):
    """Test with and without exclude last"""
    x = np.random.randn(10, 2)
    log_prob = [
        torch.ones(len(x), dtype=torch.float32),
        2 * torch.ones(len(x), dtype=torch.float32),
        3 * torch.ones(len(x), dtype=torch.float32),
    ]

    models = [
        MagicMock(spec=DummyFlow, log_prob=MagicMock(return_value=lp))
        for lp in log_prob
    ]

    n_expected = len(models)
    if exclude_last:
        n_expected -= 1

    ifm.models = torch.nn.ModuleList(models)
    ifm.models.training = True
    ifm.model.device = torch.device("cpu")
    ifm.n_models = len(models)

    out = IFM.log_prob_all(ifm, x, exclude_last=exclude_last)

    assert ifm.models.training is False
    assert out.dtype == np.float64
    assert out.shape == (10, n_expected)

    for model in models[:n_expected]:
        model.log_prob.assert_called_once()

    if exclude_last:
        models[-1].log_prob.assert_not_called()


def test_log_prob_all_exclude_last_error(ifm):
    """Assert an error is raised if there is only one flow"""
    x = np.random.randn(10, 2)

    ifm.models = torch.nn.ModuleList([DummyFlow()])
    ifm.models.training = True
    ifm.model.device = torch.device("cpu")
    ifm.n_models = 1

    with pytest.raises(RuntimeError, match=r"Cannot exclude last .*"):
        IFM.log_prob_all(ifm, x, exclude_last=True)


@pytest.mark.parametrize("i", [0, 1, 2, -1])
def test_sample_ith(ifm, i):
    """Assert the correct number of samples is drawn from the correct model"""
    n = 10
    x = torch.randn(n, 2)

    model = MagicMock(spec=DummyFlow)
    model.sample = MagicMock(return_value=x)
    model.device = torch.device("cpu")
    model.training = True
    model.eval = MagicMock()

    models = [DummyFlow(), DummyFlow(), DummyFlow()]
    models[i] = model
    ifm.models = torch.nn.ModuleList(models)
    ifm.models.training = True

    out = IFM.sample_ith(ifm, i, N=n)

    assert out.dtype == np.float64
    model.eval.assert_called_once()
    model.sample.assert_called_once_with(n)
    np.testing.assert_array_equal(out, x.numpy())


def test_sample_ith_error(ifm):
    """Assert an error is raised if no models have been added"""
    ifm.models = None
    with pytest.raises(RuntimeError, match=r"Models are not initialised yet!"):
        IFM.sample_ith(ifm, 1)


def test_save_weights(ifm):
    """Assert the correct method is called and the weights file is stored"""
    file = "w1.pt"
    ifm.weights_files = ["w0.pt"]

    def func(f):
        ifm.weights_file = f

    with patch(
        "nessai.flowmodel.importance.FlowModel.save_weights", side_effect=func
    ) as mock_save:
        IFM.save_weights(ifm, file)
    mock_save.assert_called_once_with(file)
    assert ifm.weights_files == ["w0.pt", file]


def test_load_all_weights(ifm):
    """Assert all the weights files are loaded"""
    weights_files = ["w0.pt", "w1.pt", "w2.pt"]
    weights = ["w0", "w1", "w2"]
    device = torch.device("cpu")
    ifm.weights_files = weights_files
    models = torch.nn.ModuleList(
        [
            MagicMock(spec=DummyFlow(), load_state_dict=MagicMock())
            for _ in range(len(weights_files))
        ]
    )
    config_out = [(f, device) for f in models]
    with patch(
        "nessai.flowmodel.importance.configure_model",
        side_effect=config_out,
    ) as mock_configure, patch("torch.load", side_effect=weights):
        IFM.load_all_weights(ifm)

    assert len(mock_configure.call_args_list) == 3
    assert all(m is n for m, n in zip(models, ifm.models))
    assert ifm.models.training is False

    for model, w in zip(models, weights):
        model.load_state_dict.assert_called_once_with(w)


@pytest.mark.parametrize("n", [None, 10, 16])
def test_update_weights_path(ifm, tmp_path, n):
    """Assert the list of weights files is correctly updated"""
    path = tmp_path / "outdir"
    path.mkdir()
    n_models = 15
    n_total = 16
    expected_files = []
    for i in range(n_total):
        d = path / f"level_{i}"
        d.mkdir()
        file = d / "model.pt"
        file.write_text("data")
        file = str(file)
        expected_files.append(file)

    ifm.n_models = n_models
    IFM.update_weights_path(ifm, str(path), n=n)
    n_expeceted = n or n_models
    assert ifm.weights_files == expected_files[:n_expeceted]


def test_update_weights_path_cannot_update(ifm):
    """Assert the list of weights files is correctly updated"""
    ifm.n_models = 0
    with pytest.raises(RuntimeError, match=r"n is None and .*"):
        IFM.update_weights_path(ifm, ".", n=None)


def test_update_weights_path_not_enough_files(ifm, tmp_path):
    """Assert the list of weights files is correctly updated"""
    path = tmp_path / "outdir"
    path.mkdir()
    n_models = 5
    n_total = 4
    expected_files = []
    for i in range(n_total):
        d = path / f"level_{i}"
        d.mkdir()
        file = d / "model.pt"
        file.write_text("data")
        file = str(file)
        expected_files.append(file)

    ifm.n_models = n_models
    with pytest.raises(RuntimeError, match=r".* Not enough files."):
        IFM.update_weights_path(ifm, str(path))


@pytest.mark.parametrize("weights_path", [None, "weights_directory"])
def test_resume(ifm, weights_path):
    """Assert resume method calls the correct methods"""
    model_config = dict(patience=20)
    updated_config = dict(patience=20, ftype="realnvp")
    output = "current_output"

    expected_path = weights_path or output

    ifm.output = output
    ifm.update_weights_path = MagicMock()
    ifm.load_all_weights = MagicMock()
    ifm.initialise = MagicMock()
    ifm._resume_n_models = 10

    with patch(
        "nessai.flowmodel.importance.update_model_config",
        return_value=updated_config,
    ) as mock_update:
        IFM.resume(ifm, model_config, weights_path)

    mock_update.assert_called_once_with(model_config)

    ifm.update_weights_path.assert_called_once_with(
        expected_path,
        n=10,
    )
    ifm.load_all_weights.assert_called_once_with
    ifm.initialise.assert_called_once()


def test_getstate(ifm):
    """Assert the correct keys are added and removed"""
    ifm.models = torch.nn.ModuleList([DummyFlow(), DummyFlow()])
    ifm._optimiser = MagicMock()
    ifm.model_config = dict(patience=20)
    ifm.initialised = True
    ifm.test = "value"

    out = IFM.__getstate__(ifm)

    assert "_optimiser" not in out
    assert "model_config" not in out
    assert out["models"] is None
    assert out["_resume_n_models"] == 2
    assert out["initialised"] is False
    assert out["test"] == "value"


@pytest.mark.integration_test
def test_resume_integration(tmp_path):
    output = tmp_path / "test"
    output.mkdir()

    config = dict(model_config=dict(n_inputs=2))

    ifm = IFM(output=output, config=config)

    ifm.initialise()

    n_models = 3

    for i in range(n_models):
        level_dir = output / f"level_{i}"
        level_dir.mkdir()
        ifm.add_new_flow()
        ifm.save_weights(os.path.join(level_dir, "model.pt"))

    pickled_ifm = pickle.dumps(ifm)

    resumed_ifm = pickle.loads(pickled_ifm)

    resumed_ifm.resume(config["model_config"])

    assert len(resumed_ifm.models) == n_models
