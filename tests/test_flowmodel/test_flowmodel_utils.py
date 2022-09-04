# -*- coding: utf-8 -*-
"""
Test the FlowModel utils.
"""
import pytest
from unittest.mock import patch

from nessai.flowmodel import update_config


def test_update_config_none():
    """Test update config for no input"""
    config = update_config(None)
    assert "model_config" in config


def test_update_config_invalid_type():
    """Test update config when an invalid argument is specified"""
    with pytest.raises(TypeError) as excinfo:
        update_config(False)
    assert "Must pass a dictionary" in str(excinfo.value)


def test_update_config_invalid_noise_scale():
    """Assert an error is raised if noise_scale is not a float"""
    config = {"noise_scale": "auto", "noise_type": "adaptive"}
    with pytest.raises(TypeError) as excinfo:
        update_config(config)
    assert "`noise_scale` must be a float" in str(excinfo.value)


def test_update_config_missing_noise_type():
    """Assert noise type is set if only `noise_scale` is given"""
    config = {"noise_scale": 1.0}
    out = update_config(config)
    assert out["noise_type"] == "constant"


def test_update_config_missing_noise_scale():
    """Assert an error is raised if noise_scaling is missing when noise_type is
    set.
    """
    config = {"noise_type": "constant"}
    with pytest.raises(RuntimeError) as excinfo:
        update_config(config)
    assert "`noise_scale` must be specified" in str(excinfo.value)


def test_update_config_n_neurons():
    """Assert `get_n_neurons` is called"""
    config = dict(model_config=dict(n_inputs=10))
    with patch(
        "nessai.flowmodel.utils.get_n_neurons", return_value=20
    ) as mock:
        config = update_config(config)
    mock.assert_called_once_with(n_neurons=None, n_inputs=10)
    assert config["model_config"]["n_neurons"] == 20
