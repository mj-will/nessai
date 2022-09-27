# -*- coding: utf-8 -*-
"""
Implementation of MaskedAutoregressiveFlow.
"""
import logging

from torch.nn import functional as F

from glasflow.nflows.distributions.normal import StandardNormal
from glasflow.nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from glasflow.nflows.transforms.base import CompositeTransform
from glasflow.nflows.transforms.normalization import BatchNorm
from glasflow.nflows.transforms.permutations import (
    RandomPermutation,
    ReversePermutation,
)

from .base import NFlow

logger = logging.getLogger(__name__)


class MaskedAutoregressiveFlow(NFlow):
    """Autoregressive flow with masked coupling transforms.

    Based on the implementation from nflows: \
        https://github.com/bayesiains/nflows/blob/master/nflows/flows/autoregressive.py
        but also included context features.

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
    context_features : int, optional
        Number of context (conditional) parameters.
    use_residual_blocks : bool, optional
        Use residual blocks in the MADE network.
    use_random_masks : bool, optional
        Use random masks in the MADE network.
    use_random_permutation : bool, optional
        Use a random permutation instead of the default reverse permutation.
    activation : function, optional
        Activation function implemented in torch.
    dropout_probability : float, optional
        Dropout probability used in each layer of the neural network
    batch_norm_within_layers : bool, optional
       Enable or disable batch norm within the neural network for each coupling
       transform
    batch_norm_between_layers : bool, optional
       Enable or disable batch norm between coupling transforms
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        context_features=None,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
