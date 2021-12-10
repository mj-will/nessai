# -*- coding: utf-8 -*-
"""
Various utilities for implementing normalising flows.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F

from nflows import transforms
from nflows.nn.nets import MLP as NFlowsMLP

from .distributions import MultivariateNormal
from .transforms import LULinear

logger = logging.getLogger(__name__)


def silu(x):
    """
    SiLU (Sigmoid-weighted Linear Unit) activation function.

    Also known as swish.

    Elfwing et al 2017: https://arxiv.org/abs/1702.03118v3
    """
    return torch.mul(x, torch.sigmoid(x))


def configure_model(config):
    """
    Setup the flow form a configuration dictionary.
    """
    from .realnvp import RealNVP
    from .maf import MaskedAutoregressiveFlow
    from .nsf import NeuralSplineFlow
    kwargs = {}
    flows = {
        'realnvp': RealNVP,
        'maf': MaskedAutoregressiveFlow,
        'frealnvp': RealNVP,
        'spline': NeuralSplineFlow,
        'nsf': NeuralSplineFlow,
    }
    activations = {
        'relu': F.relu,
        'tanh': F.tanh,
        'swish': silu,
        'silu': silu
    }

    k = config.get('kwargs', None)
    if k is not None:
        if 'activation' in k and isinstance(k['activation'], str):
            try:
                k['activation'] = activations[k['activation']]
            except KeyError as e:
                raise RuntimeError(f'Unknown activation function: {e}')

        if 'var' in k and 'distribution' not in k:
            k['distribution'] = MultivariateNormal([config['n_inputs']],
                                                   var=k['var'])
            k.pop('var')

        kwargs.update(k)

    if not isinstance(config['n_inputs'], int):
        raise TypeError('Number of inputs (n_inputs) must be an int')

    fc = config.get('flow', None)
    ftype = config.get('ftype', None)
    if fc is not None:
        model = fc(config['n_inputs'], config['n_neurons'], config['n_blocks'],
                   config['n_layers'], **kwargs)
    elif ftype is not None:
        if ftype.lower() not in flows:
            raise RuntimeError(f'Unknown flow type: {ftype}. Choose from:'
                               f'{flows.keys()}')
        if ((('mask' in kwargs and kwargs['mask'] is not None) or
                ('net' in kwargs and kwargs['net'] is not None)) and
                ftype.lower() not in ['realnvp', 'frealnvp']):
            raise RuntimeError('Custom masks and networks are only '
                               'supported for RealNVP')

        model = flows[ftype.lower()](config['n_inputs'], config['n_neurons'],
                                     config['n_blocks'], config['n_layers'],
                                     **kwargs)
    else:
        raise RuntimeError("Must specify either 'flow' or 'ftype'.")

    device = torch.device(config.get('device_tag', 'cpu'))
    if device != 'cpu':
        try:
            model.to(device)
        except RuntimeError as e:
            device = torch.device('cpu')
            logger.warning(
                "Could not send the normailising flow to the "
                f"specified device {config['device']} send to CPU "
                f"instead. Error raised: {e}"
            )
    logger.debug('Flow model:')
    logger.debug(model)

    model.device = device

    return model, device


def reset_weights(module):
    """Reset parameters of a given module in place.

    Uses the ``reset_parameters`` method from ``torch.nn.Module``

    Also checks the following modules from nflows

    - nflows.transforms.normalization.BatchNorm

    Parameters
    ----------
    module : :obj:`torch.nn.Module`
        Module to reset
    """
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    elif isinstance(module, transforms.BatchNorm):
        # nflows BatchNorm does not have a weight reset, so must
        # be done manually
        constant = np.log(np.exp(1 - module.eps) - 1)
        module.unconstrained_weight.data.fill_(constant)
        module.bias.data.zero_()
        module.running_mean.zero_()
        module.running_var.fill_(1)
    else:
        logger.warning(f'Could not reset: {module}')


def reset_permutations(module):
    """Resets permutations and linear transforms for a given module in place.

    Resets using the original initialisation method. This needed since they
    do not have a ``reset_parameters`` method.

    Parameters
    ----------
    module : :obj:`torch.nn.Module`
        Module to reset
    """
    if isinstance(module, (transforms.LULinear, LULinear)):
        module.cache.invalidate()
        module._initialize(identity_init=True)
    elif isinstance(module, transforms.RandomPermutation):
        module._permutation = torch.randperm(len(module._permutation))


class MLP(NFlowsMLP):
    """
    MLP which can be called with context.
    """
    def forward(self, inputs, context=None):
        """Forward method that allows for kwargs such as context.

        Parameters
        ----------
        inputs : :obj:`torch.tensor`
            Inputs to the MLP
        context : None
            Conditional inputs, must be None. Only implemented to the
            function is compatible with other methods.

        Raises
        ------
        RuntimeError
            If the context is not None.
        """
        if context is not None:
            raise NotImplementedError(
                'MLP with conditional inputs is not implemented.'
            )
        return super().forward(inputs)


def create_linear_transform(linear_transform, features):
    """Function for creating linear transforms.

    Parameters
    ----------
    linear_transform : {'permutation', 'lu', 'svd'}
        Linear transform to use.
    featres : int
        Number of features.
    """
    if linear_transform.lower() == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif linear_transform.lower() == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            LULinear(features, identity_init=True, using_cache=True)
        ])
    elif linear_transform.lower() == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(
                features, num_householder=10, identity_init=True
            )
        ])
    else:
        raise ValueError(
            f'Unknown linear transform: {linear_transform}. '
            'Choose from: {permutation, lu, svd}.'
        )
