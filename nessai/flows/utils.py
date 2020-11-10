import logging
import numpy as np
import torch
import torch.nn.functional as F

from nflows.transforms.normalization import BatchNorm
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import MLP

from .realnvp import FlexibleRealNVP
from .maf import MaskedAutoregressiveFlow
from .nsf import NeuralSplineFlow
from .distributions import MultivariateNormal

logger = logging.getLogger(__name__)


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

        if 'var' in k and 'distribution' not in k:
            k['distribution'] = MultivariateNormal([config['n_inputs']],
                                                   var=k['var'])
            k.pop('var')

        kwargs.update(k)

    if not isinstance(config['n_inputs'], int):
        raise TypeError('Number of inputs (n_inputs) must be an int')

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


def reset_weights(module):
    """
    Reset parameters of a given module in place using `reset_parameters`
    method.

    Also checks the following modules from nflows
    * nflows.transforms.normalization.BatchNorm

    Parameters
    ----------
    module : :obj:`torch.nn.Module`
        Module to reset
    """
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    elif isinstance(module, BatchNorm):
        # nflows BatchNorm does not have a weight reset, so must
        # be done manually
        constant = np.log(np.exp(1 - module.eps) - 1)
        module.unconstrained_weight.data.fill_(constant)
        module.bias.data.zero_()
        module.running_mean.zero_()
        module.running_var.fill_(1)


def reset_permutations(module):
    """
    Resets permutations and linear transforms to their original initialisation
    since these do not have a `reset_parameters` method
    """
    if isinstance(module, LULinear):
        module.cache.invalidate()
        module._initialize(identity_init=True)
    elif isinstance(module, RandomPermutation):
        module._permutation = torch.randperm(len(module._permutation))


class CustomMLP(MLP):
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
