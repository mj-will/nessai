import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import Flow, SimpleRealNVP, MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.uniform import BoxUniform
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nde.made import MaskedLinear
from nflows.utils import create_alternating_binary_mask

from nflows.distributions.base import Distribution
from nflows.utils import torchutils

from nflows import utils

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
    elif any(isinstance(m, l) for l in layers):
        m.reset_parameters()


def silu(x):
    """Silu activation function"""
    return torch.mul(x, torch.sigmoid(x))


def setup_model(config):
    """
    Setup the flow form a configuration dictionary
    """
    kwargs = {}
    flows = {'realnvp': SimpleRealNVP, 'maf': MaskedAutoregressiveFlow,
            'frealnvp': FlexibleRealNVP, 'spline': NeuralSplineFlow}
    activations = {'relu': F.relu, 'tanh': F.tanh, 'swish': silu,
            'silu': silu}

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

class TweakedUniform(BoxUniform):
    def log_prob(self, value, context=None):
        return super().log_prob(value)

    def sample(self, num_samples, context=None):
        return super().sample((num_samples, ))

    def sample_and_log_prob(self, num_samples, context=None):
        s = self.sample(num_samples)
        return s, self.log_prob(s, context)


class SimpleFlow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
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
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples
        )

        samples, logabsdet = self._transform.inverse(noise)

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
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
        num_bins=10,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        **kwargs
    ):

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
                        even=(i % 2 == 0)
                    ),
                    transform_net_create_fn=create_resnet,
                    num_bins=num_bins,
                    apply_unconditional_transform=False,
                    **kwargs
                )


        transforms = []
        for i in range(num_layers):
            transform = spline_constructor(i)
            transforms.append(transform)
            if batch_norm_between_layers:
                transforms.append(BatchNorm(features=features))

        distribution = TweakedUniform(
            low=torch.zeros(features),
            high=torch.ones(features)
        )

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=distribution,
        )


#class AugmentedCouplingTransform(Transform):
#    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
#    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
#    provided 1D mask."""
#
#    def __init__(self, features, augmented_features, transform_net_create_fn):
#        """
#        Constructor.
#        Args:
#            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
#                * If `mask[i] > 0`, `input[i]` will be transformed.
#                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
#        """
#        mask = torch.as_tensor(mask)
#        if mask.dim() != 1:
#            raise ValueError("Mask must be a 1-dim tensor.")
#        if mask.numel() <= 0:
#            raise ValueError("Mask can't be empty.")
#
#        super().__init__()
#        self.features = len(mask)
#        features_vector = torch.arange(self.features)
#
#        self.register_buffer(
#            "identity_features", features_vector.masked_select(mask <= 0)
#        )
#        self.register_buffer(
#            "transform_features", features_vector.masked_select(mask > 0)
#        )
#
#        assert self.num_identity_features + self.num_transform_features == self.features
#
#        self.transform_net = transform_net_create_fn(
#            self.num_identity_features,
#            self.num_transform_features * self._transform_dim_multiplier(),
#        )
#
#        if unconditional_transform is None:
#            self.unconditional_transform = None
#        else:
#            self.unconditional_transform = unconditional_transform(
#                features=self.num_identity_features
#            )
#
#    @property
#    def num_identity_features(self):
#        return len(self.identity_features)
#
#    @property
#    def num_transform_features(self):
#        return len(self.transform_features)
#
#    def forward(self, inputs, context=None):
#        if inputs.dim() not in [2, 4]:
#            raise ValueError("Inputs must be a 2D or a 4D tensor.")
#
#        if inputs.shape[1] != self.features:
#            raise ValueError(
#                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
#            )
#
#        identity_split = inputs[:, self.identity_features, ...]
#        transform_split = inputs[:, self.transform_features, ...]
#
#        transform_params = self.transform_net(identity_split, context)
#        transform_split, logabsdet = self._coupling_transform_forward(
#            inputs=transform_split, transform_params=transform_params
#        )
#
#        if self.unconditional_transform is not None:
#            identity_split, logabsdet_identity = self.unconditional_transform(
#                identity_split, context
#            )
#            logabsdet += logabsdet_identity
#
#        outputs = torch.empty_like(inputs)
#        outputs[:, self.identity_features, ...] = identity_split
#        outputs[:, self.transform_features, ...] = transform_split
#
#        return outputs, logabsdet
#
#    def inverse(self, inputs, context=None):
#        if inputs.dim() not in [2, 4]:
#            raise ValueError("Inputs must be a 2D or a 4D tensor.")
#
#        if inputs.shape[1] != self.features:
#            raise ValueError(
#                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
#            )
#
#        identity_split = inputs[:, self.identity_features, ...]
#        transform_split = inputs[:, self.transform_features, ...]
#
#        logabsdet = 0.0
#        if self.unconditional_transform is not None:
#            identity_split, logabsdet = self.unconditional_transform.inverse(
#                identity_split, context
#            )
#
#        transform_params = self.transform_net(identity_split, context)
#        transform_split, logabsdet_split = self._coupling_transform_inverse(
#            inputs=transform_split, transform_params=transform_params
#        )
#        logabsdet += logabsdet_split
#
#        outputs = torch.empty_like(inputs)
#        outputs[:, self.identity_features] = identity_split
#        outputs[:, self.transform_features] = transform_split
#
#        return outputs, logabsdet
#
#    def _transform_dim_multiplier(self):
#        return 2
#
#    def _scale_and_shift(self, transform_params):
#        unconstrained_scale = transform_params[:, self.num_transform_features :, ...]
#        shift = transform_params[:, : self.num_transform_features, ...]
#        # scale = (F.softplus(unconstrained_scale) + 1e-3).clamp(0, 3)
#        scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
#        return scale, shift
#
#    def _coupling_transform_forward(self, inputs, transform_params):
#        scale, shift = self._scale_and_shift(transform_params)
#        log_scale = torch.log(scale)
#        outputs = inputs * scale + shift
#        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
#        return outputs, logabsdet
#
#    def _coupling_transform_inverse(self, inputs, transform_params):
#        scale, shift = self._scale_and_shift(transform_params)
#        log_scale = torch.log(scale)
#        outputs = (inputs - shift) / scale
#        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
#        return outputs, logabsdet
#
