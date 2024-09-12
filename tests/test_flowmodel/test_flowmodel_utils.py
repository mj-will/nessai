# -*- coding: utf-8 -*-
"""
Test the FlowModel utils.
"""

from unittest.mock import patch

import pytest

from nessai.flowmodel import config as default_config
from nessai.flowmodel.utils import (
    update_config,
    update_flow_config,
    update_training_config,
)


def test_update_config():
    flow_config = dict(n_blocks=2)
    training_config = dict(patience=1)
    expected_out_flow = object()
    expected_out_training = object()
    with (
        patch(
            "nessai.flowmodel.utils.update_flow_config",
            return_value=expected_out_flow,
        ) as update_fcfg,
        patch(
            "nessai.flowmodel.utils.update_training_config",
            return_value=expected_out_training,
        ) as update_tcfg,
    ):
        flow_config_out, training_config_out = update_config(
            flow_config, training_config
        )

    update_fcfg.assert_called_once_with(flow_config)
    update_tcfg.assert_called_once_with(training_config)

    assert flow_config_out is expected_out_flow
    assert training_config_out is expected_out_training


def test_update_config_deprecated():
    """Test with a basic config"""
    model_config = dict(n_blocks=2)
    config = dict(
        patience=1,
        device_tag="cuda",
        inference_device_tag="cuda",
        model_config=model_config,
    )
    flow_config, training_config = update_config(config)
    assert flow_config["n_blocks"] == 2
    assert training_config["patience"] == 1
    assert training_config["device_tag"] == "cuda"
    assert training_config["inference_device_tag"] == "cuda"


def test_update_training_config_invalid_type():
    """Test update config when an invalid argument is specified"""
    with pytest.raises(TypeError, match="Must pass a dictionary"):
        update_training_config(False)


def test_update_training_config_none():
    assert update_training_config(None) == default_config.training.asdict()


def test_update_training_config_invalid_noise_scale():
    """Assert an error is raised if noise_scale is not a float"""
    config = {"noise_scale": "auto", "noise_type": "adaptive"}
    with pytest.raises(TypeError, match=r"`noise_scale` must be a float"):
        update_training_config(config)


def test_update_training_config_missing_noise_type():
    """Assert noise type is set if only `noise_scale` is given"""
    config = {"noise_scale": 1.0}
    out = update_training_config(config)
    assert out["noise_type"] == "constant"


def test_update_training_config_missing_noise_scale():
    """Assert an error is raised if noise_scaling is missing when noise_type is
    set.
    """
    config = {"noise_type": "constant"}
    with pytest.raises(RuntimeError, match="`noise_scale` must be specified"):
        update_training_config(config)


def test_update_flow_config_n_neurons():
    """Assert `get_n_neurons` is called"""
    config = dict(n_inputs=10)
    with patch(
        "nessai.flowmodel.utils.get_n_neurons", return_value=20
    ) as mock:
        config = update_flow_config(config)
    mock.assert_called_once_with(n_neurons=None, n_inputs=10)
    assert config["n_neurons"] == 20


def test_update_model_config_type_error():
    """Assert a type error is raised if the input is the wrong type"""
    with pytest.raises(TypeError, match="Must pass a dictionary"):
        update_flow_config(False)


def test_update_flow_config_none():
    """Test update config for no input"""
    assert update_flow_config(None) == default_config.flow.asdict()
