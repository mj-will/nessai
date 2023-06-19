# -*- coding: utf-8 -*-
"""
Implementation of Neural Spline Flows.
"""
import logging

import torch.nn.functional as F

from glasflow.nflows.distributions import StandardNormal
from glasflow.nflows import transforms
from glasflow.nflows.nn.nets import ResidualNet
from glasflow.nflows.utils import create_alternating_binary_mask

from .base import NFlow
from .utils import create_linear_transform

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
        Number of coupling transformations
    num_blocks_per_layer : int
        Number of layers (or blocks for resnet) per neural network for
        each coupling transform
    num_bins : int, optional (8)
        Number of bins to use for each spline
    context_features : int, optional
        Number of context (conditional) parameters.
    activation : function
        Activation function implemented in torch
    dropout_probability : float, optional (0.0)
        Dropout probability used in each layer of the neural network
    batch_norm_within_layers : bool, optional (False)
       Enable or disable batch norm within the neural network for each coupling
       transform
    batch_norm_between_layers : bool, optional (False)
       Enable or disable batch norm between coupling transforms
    linear_transform : {'permutation', 'lu', 'svd'}
        Linear transform to use between coupling layers. Not recommended when
        using a custom mask.
    distribution : :obj: `glasflow.nflows.distributions.Distribution`
        Distribution for the latent space.
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
        linear_transform="permutation",
        tails="linear",
        tail_bound=5.0,
        distribution=None,
        **kwargs,
    ):

        if features <= 1:
            raise ValueError(
                "Coupling based Neural Spline flow requires at least 2 "
                f"dimensions. Specified dimensions: {features}."
            )

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
                    features=features, even=(i % 2 == 0)
                ),
                transform_net_create_fn=create_resnet,
                num_bins=num_bins,
                apply_unconditional_transform=apply_unconditional_transform,
                tails=tails,
                tail_bound=tail_bound,
                **kwargs,
            )

        transforms_list = []
        for i in range(num_layers):
            if linear_transform is not None:
                transforms_list.append(
                    create_linear_transform(linear_transform, features)
                )
            transforms_list.append(spline_constructor(i))
            if batch_norm_between_layers:
                transforms_list.append(transforms.BatchNorm(features=features))

        if distribution is None:
            distribution = StandardNormal([features])

        super().__init__(
            transform=transforms.CompositeTransform(transforms_list),
            distribution=distribution,
        )
