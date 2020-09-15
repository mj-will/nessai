import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import MaskedAutoregressiveFlow as BaseMAF
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


def weight_reset(module):
    """
    Reset parameters of a given module in place

    Checks the following modules from torch.nn
    * Batchnorm1d
    * Conv1d
    * Conv2d
    * Linear

    Also checks the following modules from nflows
    * nflows.transforms.normalization.BatchNorm
    * nflows.nn.nde.made.MaskedLinear

    Parameters
    ----------
    module : :obj:`torch.nn.Module`
        Module to reset
    """
    layers = [nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, MaskedLinear]
    if isinstance(module, BatchNorm):
        # nflows BatchNorm does not have a weight reset, so must
        # be done manually
        constant = np.log(np.exp(1 - module.eps) - 1)
        module.unconstrained_weight.data.fill_(constant)
        module.bias.data.zero_()
        module.running_mean.zero_()
        module.running_var.fill_(1)
    elif any(isinstance(module, layer) for layer in layers):
        module.reset_parameters()


def silu(x):
    """
    SiLU (Sigmoid-weighted Linear Unit) activation function.

    Also known as swish.

    Elfwing et al 2017: https://arxiv.org/abs/1702.03118v3
    """
    return torch.mul(x, torch.sigmoid(x))


def setup_model(config):
    """
    Setup the flow form a configuration dictionary.
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


class Flow(Distribution):
    """
    Base class for all flow objects.

    This replaces `Flow` from nflows. It removes the context and includes
    additional methods which are called in FlowModel.

    Parameters
    ----------
    transform : :obj: `nflows.transforms.Transform`
        Object that applys the transformation, must have`forward` and
        `inverse` methods. See nflows for more details.
    distribution : :obj: `nflows.distributions.Distribution`
        Object the serves as the base distribution used when sampling
        and computing the log probrability. Must have `log_prob` and
        `sample` methods. See nflows for details
    """
    def __init__(self, transform, distribution):
        super().__init__()

        for method in ['forward', 'inverse']:
            if not hasattr(transform, method):
                raise RuntimeError(
                    f'Transform does not have `{method}` method')

        for method in ['log_prob', 'sample']:
            if not hasattr(distribution, method):
                raise RuntimeError(
                    f'Distribution does not have `{method}` method')

        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, inputs, context=None):
        """
        Computes the log probability of the inputs samples by apply the
        transform.

        Does NOT need to specified by the user
        """
        noise, logabsdet = self._transform(inputs)
        log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def _sample(self, num_samples, context=None):
        """
        Produces N samples in the data space by drawing from the base
        distribution and the applying the inverse transform.

        Does NOT need to be specified by the user
        """
        noise = self._distribution.sample(num_samples)

        samples, _ = self._transform.inverse(noise)

        return samples

    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the latent
        space and log |J|
        """
        return self._transform.forward(x)

    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log |J| (not log probability)
        """
        return self._transform.inverse(z)

    def sample_and_log_prob(self, N, context=None):
        """
        Generates samples from the flow, together with their log probabilities
        in the data space log p(x) = log p(z) + log|J|.

        For flows, this is more efficient that calling `sample` and `log_prob`
        separately.
        """
        z, log_prob = self._distribution.sample_and_log_prob(N)

        samples, logabsdet = self._transform.inverse(z)

        return samples, log_prob - logabsdet

    def base_distribution_log_prob(self, z, context=None):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow.
        """
        return self._distribution.log_prob(z)


class CustomMLP(nets.MLP):
    """
    MLP which handles additional kwargs that are supplied by some
    flow models
    """
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

    Parameters
    ----------
    features : int
        Number of features (dimensions) in the data space
    hidden_features : int
        Number of neurons per layer in each neural network
    num_layers : int
        Number of coupling tranformations
    num_blocks_per_layer : int
        Number of layers (or blocks for resnet) per nerual network for
        each coupling transform
    mask : array_like, optional
        Custom mask to use between coupling transforms. Can either be
        a single array with the same length as the number of features or
        and two-dimensional array of shape (# features, # num_layers).
        Must use -1 and 1 to indicate no updated and updated.
    net : {'resnet', 'mlp'}
        Type of neural network to use
    use_volume_preserving : bool, optional (False)
        Use volume preserving flows which use only additiona and no scaling
    activation : function
        Activation function implemented in torch
    dropout_probability : float, optional (0.0)
        Dropout probaiblity used in each layer of the neural network
    batch_norm_within_layers : bool, optional (False)
       Enable or disable batch norm within the neural network for each coupling
       transform
    batch_norm_between_layers : bool, optional (False)
       Enable or disable batch norm between coupling transforms
    linear_transform : {'permutaiton', 'lu', 'svd'}
        Linear transform to use between coupling layers. Not recommended when
        using a custom mask.
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

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )


class MaskedAutoregressiveFlow(BaseMAF):
    """
    Wrapper for MaskedAutoregressiveFlow included in nflows that adds
    additional methods that are used in FlowModel.

    See: https://github.com/bayesiains/nflows/blob/master/nflows/flows/
    autoregressive.py
    """

    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the latent
        space and log |J|
        """
        return self._transform.forward(x)

    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log |J|
        """
        return self._transform.inverse(z, context=context)

    def base_distribution_log_prob(self, z, context=None):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow
        """
        return self._distribution.log_prob(z, context=context)


class NeuralSplineFlow(Flow):
    """
    Implementation of Neural Spline Flow

    NSF: https://arxiv.org/abs/1906.04032

    Parameters
    ----------
    features : int
        Number of features (dimensions) in the data space
    hidden_features : int
        Number of neurons per layer in each neural network
    num_layers : int
        Number of coupling tranformations
    num_blocks_per_layer : int
        Number of layers (or blocks for resnet) per nerual network for
        each coupling transform
    num_bins : int, optional (8)
        Number of bins to use for each spline
    activation : function
        Activation function implemented in torch
    dropout_probability : float, optional (0.0)
        Dropout probaiblity used in each layer of the neural network
    batch_norm_within_layers : bool, optional (False)
       Enable or disable batch norm within the neural network for each coupling
       transform
    batch_norm_between_layers : bool, optional (False)
       Enable or disable batch norm between coupling transforms
    linear_transform : {'permutaiton', 'lu', 'svd'}
        Linear transform to use between coupling layers. Not recommended when
        using a custom mask.
    kwargs : dict
        Additional kwargs parsed to the spline constructor, e.g. `tails` or
        `tail_bound`. See nflows for details
    """
    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        num_bins=8,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        apply_unconditional_transform=False,
        linear_transform='permutation',
        **kwargs
    ):
        if batch_norm_between_layers:
            logger.warning('BatchNorm cannot be used with splines')
            batch_norm_between_layers = False

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

        distribution = StandardNormal([features])

        super().__init__(
            transform=CompositeTransform(transforms_list),
            distribution=distribution,
        )
