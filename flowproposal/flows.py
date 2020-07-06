import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import SimpleRealNVP, MaskedAutoregressiveFlow

logger = logging.getLogger(__name__)


def swish(x):
    """Swish activation function"""
    return torch.mul(x, torch.sigmoid(x))


def setup_model(config):
    """
    Setup the flow form a configuration dictionary
    """
    kwargs = {}
    flows = {'realnvp': SimpleRealNVP, 'maf': MaskedAutoregressiveFlow}
    activations = {'relu': F.relu, 'tanh': F.tanh, 'swish': swish}

    if 'kwargs' in config and (k:=config['kwargs']) is not None:
        if 'activation' in k and isinstance(k['activation'], str):
            try:
                k['activation'] = activations[k['activatiob']]
            except KeyError as e:
                raise RuntimeError(f'Unknown activation function {e}')

        kwargs.update(k)

    if 'flow' in config and (c:=config['flow']) is not None:
        model = c(config['n_inputs'], config['n_neurons'], config['n_blocks'],
                config['n_layers'], **kwargs)
    elif 'ftype' in config and (f:=config['ftype']) is not None:
        if f.lower() not in flows:
            raise RuntimeError((f'Unknown flow type: {f}. Choose from:'
                f'{flows.keys()}'))

        model = flows[f.lower()](config['n_inputs'], config['n_neurons'], config['n_blocks'],
                config['n_layers'], **kwargs)

    if 'device_tag' in config:
        if isinstance(config['device_tag'], str):
            device = torch.device(config['device_tag'])

        try:
            model.to(device)
        except RuntimeError as e:
            device = torch.device('cpu')
            logger.warning(('Could not send the normailising flow to the specified'
                f"device {config['device']} send to CPU instead. Error raised: {e}"))
    logger.debug('Flow model:')
    logger.debug(model)

    model.device = device

    return model, device
