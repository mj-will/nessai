import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import Flow, MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows import transforms
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    )
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nde.made import MaskedLinear
from nflows.utils import create_alternating_binary_mask
from nflows.distributions.base import Distribution

logger = logging.getLogger(__name__)


def weight_reset(m):
    """
    Reset parameters of a given model
    """
    layers = [nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, MaskedLinear]
    if isinstance(m, BatchNorm):
        # nflows BatchNorm does not have a weight reset, so must
        # be done manually
        constant = np.log(np.exp(1 - m.eps) - 1)
        m.unconstrained_weight.data.fill_(constant)
        m.bias.data.zero_()
        m.running_mean.zero_()
        m.running_var.fill_(1)
    elif any(isinstance(m, layer) for layer in layers):
        m.reset_parameters()


def silu(x):
    """Silu activation function"""
    return torch.mul(x, torch.sigmoid(x))


def setup_model(config):
    """
    Setup the flow form a configuration dictionary
    """
    kwargs = {}
    flows = {'realnvp': FlexibleRealNVP, 'maf': MaskedAutoregressiveFlow,
             'frealnvp': FlexibleRealNVP, 'spline': NeuralSplineFlow}
    activations = {'relu': F.relu, 'tanh': F.tanh, 'swish': silu, 'silu': silu}

    if 'kwargs' in config and (k := config['kwargs']) is not None:
        if 'activation' in k and isinstance(k['activation'], str):
            try:
                k['activation'] = activations[k['activation']]
            except KeyError as e:
                raise RuntimeError(f'Unknown activation function {e}')

        kwargs.update(k)

    if 'flow' in config and (c := config['flow']) is not None:
        model = c(config['n_inputs'], config['n_neurons'], config['n_blocks'],
                  config['n_layers'], **kwargs)
    elif 'ftype' in config and (f := config['ftype']) is not None:
        if f.lower() not in flows:
            raise RuntimeError(f'Unknown flow type: {f}. Choose from:'
                               f'{flows.keys()}')
        if ('mask' in kwargs and kwargs['mask'] is not None) or \
                ('net' in kwargs and kwargs['net'] is not None):
            if f not in ['realnvp', 'frealnvp']:
                raise RuntimeError('Custom masks and networks are only '
                                   'supported for RealNVP')

        model = flows[f.lower()](config['n_inputs'], config['n_neurons'],
                                 config['n_blocks'], config['n_layers'],
                                 **kwargs)

    if 'device_tag' in config:
        if isinstance(config['device_tag'], str):
            device = torch.device(config['device_tag'])

        try:
            model.to(device)
        except RuntimeError as e:
            device = torch.device('cpu')
            logger.warning("Could not send the normailising flow to the "
                           f"specified device {config['device']} send to CPU "
                           f"instead. Error raised: {e}")
    logger.debug('Flow model:')
    logger.debug(model)

    model.device = device

    return model, device


class CustomMLP(nets.MLP):

    def __init__(self, *args, **kwargs):
        super(CustomMLP, self).__init__(*args, **kwargs)

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
        linear_transform=None
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

        def create_linear_transform():
            if linear_transform == 'permutation':
                return transforms.RandomPermutation(features=features)
            elif linear_transform == 'lu':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=features),
                    transforms.LULinear(features, identity_init=True)
                ])
            elif linear_transform == 'svd':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=features),
                    transforms.SVDLinear(features, num_householder=10,
                                         identity_init=True)
                ])
            else:
                raise ValueError

        if net.lower() == 'resnet':
            def create_net(in_features, out_features):
                return nets.ResidualNet(
                    in_features,
                    out_features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers
                    )
        elif net.lower() == 'mlp':
            if batch_norm_within_layers:
                logger.warning('Batch norm within layers not supported for '
                               'MLP, will be ignored')
            if dropout_probability:
                logger.warning('Dropout not supported for MLP, '
                               'will be ignored')
            hidden_features = num_blocks_per_layer * [hidden_features]

            def create_net(in_features, out_features):
                return CustomMLP(
                        (in_features,),
                        (out_features,),
                        hidden_features,
                        activation=activation)

        else:
            raise RuntimeError(f'Unknown nn type: {net}')

        layers = []
        for i in range(num_layers):
            if linear_transform is not None:
                layers.append(create_linear_transform())
            transform = coupling_constructor(
                mask=mask[i], transform_net_create_fn=create_net
            )
            layers.append(transform)
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        if linear_transform is not None:
            layers.append(create_linear_transform())

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )


class SimpleFlow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution):
        super().__init__()
        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, inputs, context=None):
        noise, logabsdet = self._transform(inputs)
        log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def _sample(self, num_samples, context=None):
        noise = self._distribution.sample(num_samples)

        samples, _ = self._transform.inverse(noise)

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """
        Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob`
        separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples
        )

        samples, logabsdet = self._transform.inverse(noise)

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """
        Transforms given data into noise. Useful for goodness-of-fit
        checking.
        """
        noise, _ = self._transform(inputs)
        return noise


class NeuralSplineFlow(SimpleFlow):

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        num_bins=64,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        apply_unconditional_transform=False,
        linear_transform_type='permutation',
        **kwargs
    ):
        if batch_norm_between_layers:
            logger.warning('BatchNorm cannot be used with splines')
            batch_norm_between_layers = False

        def create_linear_transform():
            if linear_transform_type == 'permutation':
                return transforms.RandomPermutation(features=features)
            elif linear_transform_type == 'lu':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=features),
                    transforms.LULinear(features, identity_init=True)
                ])
            elif linear_transform_type == 'svd':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=features),
                    transforms.SVDLinear(features, num_householder=10,
                                         identity_init=True)
                ])
            else:
                raise ValueError

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

        def spline_constructor(i):
            return PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    features=features,
                    even=(i % 2 == 0)),
                transform_net_create_fn=create_resnet,
                num_bins=num_bins,
                apply_unconditional_transform=apply_unconditional_transform,
                **kwargs)

        transforms_list = []
        for i in range(num_layers):
            transforms_list.append(create_linear_transform())
            transforms_list.append(spline_constructor(i))

        transforms_list.append(create_linear_transform())
        distribution = StandardNormal([features])

        super().__init__(
            transform=CompositeTransform(transforms_list),
            distribution=distribution,
        )
