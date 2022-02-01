# -*- coding: utf-8 -*-
"""
Various utilities for implementing normalising flows.
"""
import inspect
import logging
import numpy as np
import torch
import torch.nn.functional as F

from nflows import transforms
from nflows.nn.nets import MLP as NFlowsMLP

from .distributions import MultivariateNormal, ResampledGaussian
from .transforms import LULinear

logger = logging.getLogger(__name__)


def silu(x):
    """
    SiLU (Sigmoid-weighted Linear Unit) activation function.

    Also known as swish.

    Elfwing et al 2017: https://arxiv.org/abs/1702.03118v3
    """
    return torch.mul(x, torch.sigmoid(x))


def get_base_distribution(
    n_inputs, distribution, **kwargs
):
    """Get the base distribution for a flow.

    Includes special configuration for certain distributions.

    Parameters
    ----------
    n_inputs : int
        Number of inputs to the distribution.
    """
    distributions = {
        'lars': ResampledGaussian,
        'resampled_gaussian': ResampledGaussian,
        'mvn': MultivariateNormal,
    }

    dist_class = None

    if isinstance(distribution, str):
        dist_class = distributions.get(distribution.lower())
        if not dist_class:
            raise ValueError(
                f'Unknown distribution: {distribution}'
            )
    elif inspect.isclass(distribution):
        logger.debug('Distribution is class. Creating an instance.')
        dist_class = distribution

    if dist_class:
        logger.debug('Creating instance of the base distribution')
        if dist_class is ResampledGaussian:
            n_layers = kwargs.pop('n_layers', 2)
            n_neurons = kwargs.pop('n_neurons', 128)
            layers_list = n_layers * [n_neurons]
            logger.debug(
                f'LARS acceptance network will have {n_layers} layers with '
                f'{n_neurons} neurons each.'
            )
            acc_fn = MLP(
                [n_inputs], [1], layers_list, activate_output=torch.sigmoid
            )
            dist = dist_class([n_inputs], acc_fn, **kwargs)
        else:
            dist = dist_class([n_inputs], **kwargs)
    elif distribution is None:
        dist = None
    else:
        dist = distribution

    return dist


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

    if not isinstance(config['n_inputs'], int):
        raise TypeError('Number of inputs (n_inputs) must be an int')

    k = config.get('kwargs', None)
    if k is not None:
        if 'activation' in k and isinstance(k['activation'], str):
            try:
                k['activation'] = activations[k['activation']]
            except KeyError as e:
                raise RuntimeError(f'Unknown activation function: {e}')

        kwargs.update(k)

    dist_kwargs = config.pop('distribution_kwargs', None)
    if dist_kwargs is None:
        dist_kwargs = {}
    distribution = get_base_distribution(
        config['n_inputs'],
        config.pop('distribution', None),
        **dist_kwargs,
    )
    # Allows for classes that can't handle distribution as an input
    if distribution:
        kwargs['distribution'] = distribution

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
    """Wrapper for the MLP included in nflows.

    Adds option to handle :code:`context=None` and using a different activation
    function for the output layer.

    See the implementation in nflows for details: \
        https://github.com/bayesiains/nflows/blob/master/nflows/nn/nets/mlp.py#L9
    """
    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_sizes,
        activation=F.leaky_relu,
        activate_output=False,
    ):
        super().__init__(
            in_shape,
            out_shape,
            hidden_sizes,
            activation,
            False
        )

        if activate_output:
            self.activate_output = True
            if activate_output is True:
                self._output_activation = self._activation
            elif callable(activate_output):
                self._output_activation = activate_output
            else:
                raise ValueError(
                    'activate_output must be a boolean or a callable'
                )
        else:
            self.activate_output = False

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
        out = super().forward(inputs)
        if self.activate_output:
            out = self._output_activation(out)
        return out


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
