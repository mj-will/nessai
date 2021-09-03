# -*- coding: utf-8 -*-
"""
Implementation of Non-Linear Squared Flow
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from nflows.distributions import StandardNormal
from nflows import transforms
from nflows.nn.nets import ResidualNet

from .base import NFlow
from .transforms import NLSqCouplingTransform

logger = logging.getLogger(__name__)


class NonLinearSquaredFlow(NFlow):
    """Implementation of Non-linear Squared Flow.

    Reference: Ziegler & Rush 2019, arXiv:1901.10548

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
    activation : function
        Activation function implemented in torch
    dropout_probability : float, optional
        Dropout probaiblity used in each layer of the neural network
    batch_norm_within_layers : bool, optional (False)
       Enable or disable batch norm within the neural network for each coupling
       transform
    batch_norm_between_layers : bool, optional
       Enable or disable batch norm between coupling transforms
    linear_transform : {'permutation', 'lu', 'svd', None}
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
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        linear_transform=None,
        distribution=None,
        alpha=0.95,
    ):

        coupling_constructor = NLSqCouplingTransform

        if mask is None:
            mask = torch.ones(features)
            mask[::2] = -1
        else:
            mask = np.array(mask)
            if not mask.shape[-1] == features:
                raise ValueError('Mask does not match number of features')
            if mask.ndim == 2 and not mask.shape[0] == num_layers:
                raise ValueError('Mask does not match number of layers')

            mask = torch.from_numpy(mask).type(torch.get_default_dtype())

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
                    transforms.LULinear(features, identity_init=True,
                                        using_cache=True)
                ])
            elif linear_transform == 'svd':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=features),
                    transforms.SVDLinear(features, num_householder=10,
                                         identity_init=True)
                ])
            else:
                raise ValueError(
                    f'Unknown linear transform: {linear_transform}. '
                    'Choose from: {permutation, lu, svd, None}.'
                )

        def create_net(in_features, out_features):
            return ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers
                )

        layers = []
        for i in range(num_layers):
            if linear_transform is not None:
                layers.append(create_linear_transform())
            transform = coupling_constructor(
                mask=mask[i], transform_net_create_fn=create_net, alpha=alpha
            )
            layers.append(transform)
            if batch_norm_between_layers:
                layers.append(transforms.BatchNorm(features=features))

        if distribution is None:
            distribution = StandardNormal([features])

        super().__init__(
            transform=transforms.CompositeTransform(layers),
            distribution=distribution,
        )
