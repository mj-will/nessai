# -*- coding: utf-8 -*-
"""
Implementation of Real Non Volume Preserving flows.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from glasflow.nflows.distributions import StandardNormal
from glasflow.nflows import transforms

from .base import NFlow
from .utils import create_linear_transform, create_pre_transform

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
    context_features : int, optional
        Number of context (conditional) parameters.
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
    pre_transform : str
        Linear transform to use before the first transform.
    pre_transform_kwargs : dict
        Keyword arguments to pass to the transform class used for the pre-
        transform.
    actnorm : bool
        Include activation normalisation as described in arXiv:1807.03039.
        Batch norm between layers must be disabled if using this option.
    kwargs :
        Keyword arguments are passed to the coupling class.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        mask=None,
        context_features=None,
        net="resnet",
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        linear_transform=None,
        pre_transform=None,
        pre_transform_kwargs=None,
        actnorm=False,
        distribution=None,
        **kwargs,
    ):

        if features <= 1:
            raise ValueError(
                "RealNVP requires at least 2 dimensions. "
                f"Specified dimensions: {features}."
            )

        if actnorm and batch_norm_between_layers:
            raise RuntimeError(
                "Cannot enable actnorm and batchnorm between layers "
                "simultaneously."
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
                raise ValueError("Mask does not match number of features")
            if mask.ndim == 2 and not mask.shape[0] == num_layers:
                raise ValueError("Mask does not match number of layers")

            mask = torch.from_numpy(mask).type(torch.get_default_dtype())

        if mask.dim() == 1:
            mask_array = torch.empty([num_layers, features])
            for i in range(num_layers):
                mask_array[i] = mask
                mask *= -1
            mask = mask_array

        if net.lower() == "resnet":
            from glasflow.nflows.nn.nets import ResidualNet

            def create_net(in_features, out_features):
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

        elif net.lower() == "mlp":
            from .nets import MLP

            if batch_norm_within_layers:
                logger.warning(
                    "Batch norm within layers not supported for "
                    "MLP, will be ignored"
                )
            if dropout_probability:
                logger.warning(
                    "Dropout not supported for MLP, " "will be ignored"
                )
            hidden_features = num_blocks_per_layer * [hidden_features]

            def create_net(in_features, out_features):
                return MLP(
                    (in_features,),
                    (out_features,),
                    hidden_features,
                    activation=activation,
                )

        else:
            raise ValueError(
                f"Unknown nn type: {net}. " "Choose from: {resnet, mlp}."
            )

        layers = []

        if pre_transform is not None:
            if pre_transform_kwargs is None:
                pre_transform_kwargs = {}
            layers.append(
                create_pre_transform(
                    pre_transform, features, **pre_transform_kwargs
                )
            )

        for i in range(num_layers):

            if actnorm:
                layers.append(
                    transforms.normalization.ActNorm(features=features)
                )

            if linear_transform is not None:
                layers.append(
                    create_linear_transform(linear_transform, features)
                )
            transform = coupling_constructor(
                mask=mask[i],
                transform_net_create_fn=create_net,
                **kwargs,
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
