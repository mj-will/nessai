# -*- coding: utf-8 -*-
"""
Implementation of Real Non Volume Preserving flows.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from nflows.distributions import StandardNormal
from nflows import transforms

from .base import NFlow

logger = logging.getLogger(__name__)


class RealNVP(NFlow):
    """
    Implementation of RealNVP.

    This class modifies ``SimpleRealNVP`` from nflows to allows for a custom
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
        Number of coupling transformations
    num_blocks_per_layer : int
        Number of layers (or blocks for resnet) per neural network for
        each coupling transform
    mask : array_like, optional
        Custom mask to use between coupling transforms. Can either be
        a single array with the same length as the number of features or
        and two-dimensional array of shape (# features, # num_layers).
        Must use -1 and 1 to indicate no updated and updated.
    net : {'resnet', 'mlp'}
        Type of neural network to use
    use_volume_preserving : bool, optional (False)
        Use volume preserving flows which use only addition and no scaling
    activation : function
        Activation function implemented in torch
    dropout_probability : float, optional (0.0)
        Dropout probability used in each layer of the neural network
    batch_norm_within_layers : bool, optional (False)
       Enable or disable batch norm within the neural network for each coupling
       transform
    batch_norm_between_layers : bool, optional (False)
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
        net='resnet',
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        linear_transform=None,
        distribution=None,
    ):

        if features <= 1:
            raise ValueError(
                'RealNVP requires at least 2 dimensions. '
                f'Specified dimensions: {features}.'
            )

        if use_volume_preserving:
            coupling_constructor = transforms.AdditiveCouplingTransform
        else:
            coupling_constructor = transforms.AffineCouplingTransform

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

        if net.lower() == 'resnet':
            from nflows.nn.nets import ResidualNet

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
        elif net.lower() == 'mlp':
            from .utils import MLP
            if batch_norm_within_layers:
                logger.warning('Batch norm within layers not supported for '
                               'MLP, will be ignored')
            if dropout_probability:
                logger.warning('Dropout not supported for MLP, '
                               'will be ignored')
            hidden_features = num_blocks_per_layer * [hidden_features]

            def create_net(in_features, out_features):
                return MLP(
                        (in_features,),
                        (out_features,),
                        hidden_features,
                        activation=activation)

        else:
            raise ValueError(
                f'Unknown nn type: {net}. '
                'Choose from: {resnet, mlp}.'
            )

        layers = []
        for i in range(num_layers):
            if linear_transform is not None:
                layers.append(create_linear_transform())
            transform = coupling_constructor(
                mask=mask[i], transform_net_create_fn=create_net
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
