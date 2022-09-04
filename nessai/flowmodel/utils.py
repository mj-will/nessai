# -*- coding: utf-8 -*-
"""
Utilities for configuring FlowModel.
"""
from .config import DEFAULT_MODEL_CONFIG, DEFAULT_FLOW_CONFIG
from ..flows.utils import get_n_neurons


def update_config(d):
    """
    Update the configuration dictionary to include the defaults.

    Notes
    -----
    The default configuration is specified in :py:mod:`nessai.flowmodel.config`


    The kwargs can contain any additional keyword arguments that are specific
    to the type of flow being used.

    Parameters
    ----------
    d : dict
        Dictionary with configuration

    Returns
    -------
    dict
        Dictionary with updated default configuration
    """
    default = DEFAULT_FLOW_CONFIG.copy()
    default_model = DEFAULT_MODEL_CONFIG.copy()

    if d is None:
        default["model_config"] = default_model
    else:
        if not isinstance(d, dict):
            raise TypeError(
                "Must pass a dictionary to update the default "
                "trainer settings"
            )
        else:
            default.update(d)
            default_model.update(d.get("model_config", {}))
            default_model["n_neurons"] = get_n_neurons(
                n_neurons=default_model.get("n_neurons"),
                n_inputs=default_model.get("n_inputs"),
            )
            default["model_config"] = default_model

    if default["noise_type"] is not None and default["noise_scale"] is None:
        raise RuntimeError(
            "`noise_scale` must be specified when `noise_type` is given."
        )
    if isinstance(default["noise_scale"], float):
        if default["noise_type"] is None:
            default["noise_type"] = "constant"
    elif default["noise_scale"] is not None:
        raise TypeError(
            "`noise_scale` must be a float. "
            f"'Got type: {type(default['noise_scale'])}"
        )

    return default
