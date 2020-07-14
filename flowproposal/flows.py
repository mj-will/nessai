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
            'frealnvp': FlexibleRealNVP}
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
        if ('mask' in kwargs and kwargs['mask'] is not None) or \
                ('net' in kwargs and kwargs['net'] is not None):
            if f.lower() == 'frealnvp':
                pass
            elif f.lower() == 'realnvp':
                f = 'frealnvp'
            else:
                raise RuntimeError(f'Custom masks and networks are only supported for RealNVP')

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


class ContextMLP(nets.MLP):

    def __init__(self, *args, **kwargs):
        super(ContextMLP, self).__init__(*args, **kwargs)

    def forward(self, inputs, *args, **kwargs):
        """Forward method that allows for kwargs such as context"""
        if inputs.shape[1:] != self._in_shape:
            raise ValueError(
                "Expected inputs of shape {}, got {}.".format(
                    self._in_shape, inputs.shape[1:]
                )
            )

        inputs = inputs.reshape(-1, np.prod(self._in_shape))
        outputs = self._input_layer(inputs)
        outputs = self._activation(outputs)

        for hidden_layer in self._hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self._activation(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)

        return outputs


class FlexibleRealNVP(Flow):
    """
    Modified version of SimpleRealNVP from nflows that allows for a custom
    mask to be parsed as a numpy array and allows for MLP to be used
    instead of a ResNet

    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        mask=None,
        net='resnet',
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
            if not mask.shape[-1] == features:
                raise RuntimeError('Mask does not match number of features')
            if mask.ndim == 2 and not mask.shape[0] == num_layers:
                raise RuntimeError('Mask does not match number of layers')

            mask = torch.from_numpy(mask.astype('float32'))

        if mask.dim() == 1:
            mask_array = torch.empty([num_layers, features])
            for i in range(num_layers):
                mask_array[i] = mask
                mask *= -1
            mask = mask_array

        if net.lower() == 'resnet':
            def create_net(in_features, out_features):
                return nets.ResidualNet(
                    in_features,
                    out_features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
            )
        elif net.lower() == 'mlp':
            if batch_norm_within_layers:
                logger.warning('Batch norm within layers not supported for MLP, will be ignored')
            if dropout_probability:
                logger.warning('Dropour not supported for MLP, will be ignored')
            hidden_features = num_blocks_per_layer * [hidden_features]
            def create_net(in_features, out_features):
                return ContextMLP(
                        (in_features,),
                        (out_features,),
                        hidden_features,
                        activation=activation)

        else:
            raise RuntimeError(f'Unknown nn type: {net}')

        layers = []
        for i in range(num_layers):
            transform = coupling_constructor(
                mask=mask[i], transform_net_create_fn=create_net
            )
            layers.append(transform)
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
