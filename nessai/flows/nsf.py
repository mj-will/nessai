# -*- coding: utf-8 -*-
"""
Implementation of Neural Spline Flows.
"""
import logging


import torch
import torch.nn.functional as F

from nflows.distributions import StandardNormal
from nflows import transforms
from nflows.nn.nets import ResidualNet
from nflows.utils import create_alternating_binary_mask

from .base import NFlow
from .distributions import BoxUniform

logger = logging.getLogger(__name__)


class NeuralSplineFlow(NFlow):
    """
    Implementation of Neural Spline Flow

    See: https://arxiv.org/abs/1906.04032

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
        context_features=None,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        apply_unconditional_transform=False,
        linear_transform='permutation',
        tails='linear',
        tail_bound=5.0,
        base_distribution=None,
        **kwargs
    ):

        if base_distribution == 'uniform' and tails:
            logger.warning('Tails not used with uniform distribution.')
            tails = None
            tail_bound = 1.0

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
            elif not linear_transform:
                return None
            else:
                raise ValueError

        def create_resnet(in_features, out_features):
            return ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        def spline_constructor(i):
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    features=features,
                    even=(i % 2 == 0)),
                transform_net_create_fn=create_resnet,
                num_bins=num_bins,
                apply_unconditional_transform=apply_unconditional_transform,
                tails=tails,
                tail_bound=tail_bound,
                **kwargs)

        transforms_list = []
        for i in range(num_layers):
            if linear_transform is not None:
                transforms_list.append(create_linear_transform())
            transforms_list.append(spline_constructor(i))
            if batch_norm_between_layers:
                transforms_list.append(transforms.BatchNorm(features=features))

        if base_distribution is None or base_distribution == 'gaussian':
            distribution = StandardNormal([features])
        elif base_distribution == 'uniform':
            lower = torch.zeros(features)
            upper = torch.ones(features)
            distribution = BoxUniform(lower, upper)

        super().__init__(
            transform=transforms.CompositeTransform(transforms_list),
            distribution=distribution,
        )
