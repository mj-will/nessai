import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import Flow, SimpleRealNVP, MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm

logger = logging.getLogger(__name__)


def swish(x):
    """Swish activation function"""
    return torch.mul(x, torch.sigmoid(x))


def setup_model(config):
    """
    Setup the flow form a configuration dictionary
    """
    kwargs = {}
    flows = {'realnvp': SimpleRealNVP, 'maf': MaskedAutoregressiveFlow,
            'cmrealnnvp': CustomMaskRealNVP}
    activations = {'relu': F.relu, 'tanh': F.tanh, 'swish': swish}

    if 'kwargs' in config and (k:=config['kwargs']) is not None:
        if 'activation' in k and isinstance(k['activation'], str):
            try:
                k['activation'] = activations[k['activation']]
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

        if 'mask' in kwargs and (m:=kwargs['mask']) is not None:
            if f.lower() == 'cmrealnvp':
                pass
            elif f.lower() == 'realnvp':
                f = 'cmrealnvp'
            else:
                raise RuntimeError(f'Custom masks are only supported for RealNVP')

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




class CustomMaskRealNVP(Flow):
    """
    Modified version of SimpleRealNVP from nflows that allows for a
    mask to be parsed as a numpy array

    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        mask=None,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        if mask is None:
            mask = torch.ones(features)
            mask[::2] = -1
        else:
            mask = torch.from_numpy(mask.astype('float32'))

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
