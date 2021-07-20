# -*- coding: utf-8 -*-
"""
Implementation of Neural Spline Flows.
"""
import logging

import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import torch.nn.functional as F

from .base import PyroFlow

logger = logging.getLogger(__name__)


class NeuralSplineFlow(PyroFlow):
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
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        apply_unconditional_transform=False,
        linear_transform='permutation',
        tails='linear',
        tail_bound=5.0,
        **kwargs
    ):
        def create_linear_transform():
            if linear_transform == 'permutation':
                return T.Permute(torch.randperm(features, dtype=torch.long))
            else:
                raise ValueError

        transforms = []
        for n in range(num_layers):
            transforms.append(
                T.spline_coupling(
                    features,
                    hidden_dims=num_blocks_per_layer * [hidden_features],
                    count_bins=num_bins,
                    bound=tail_bound,
                )
            )
            if linear_transform:
                transforms.append(create_linear_transform())

        base_dist = dist.Normal(torch.zeros(features), torch.ones(features))
        super().__init__(base_dist, transforms)
