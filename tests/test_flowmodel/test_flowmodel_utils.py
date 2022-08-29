# -*- coding: utf-8 -*-
"""
Test the FlowModel utils.
"""
import pytest

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


@pytest.mark.parametrize("noise_scale", ["auto", 4])
def test_update_config_invalid_noise_scale(noise_scale):
    """Assert an error is raised if noise_scale is not a float or adapative."""
    config = {"noise_scale": noise_scale}
    with pytest.raises(ValueError) as excinfo:
        update_config(config)
    assert "noise_scale must be a float or" in str(excinfo.value)


def test_update_config_n_neurons():
    """Assert the n_neurons is set to 2x n_inputs"""
    config = dict(model_config=dict(n_inputs=10))
    config = update_config(config)
    assert config["model_config"]["n_neurons"] == 20
