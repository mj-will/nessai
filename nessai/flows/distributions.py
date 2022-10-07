# -*- coding: utf-8 -*-
"""
Distributions to use as the 'base distribution' for normalising flows.
"""
import math

from glasflow.distributions import (
    ResampledGaussian as BaseResampledGaussian,
)
from glasflow.nflows.distributions import Distribution
from glasflow.nflows.utils import torchutils
import numpy as np
import torch


class MultivariateNormal(Distribution):
    """
    A multivariate Normal with zero mean and specified covariance.

    Parameters
    ----------
    shape : tuple
        Shape of distribution, this is used to determine the number of
        dimensions.
    var : float, optional
        Variance of the distribution.
    """

    def __init__(self, shape, var=1):
        super().__init__()
        self._shape = torch.Size(shape)
        self._var = var
        self._std = math.sqrt(var)

        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5 * np.prod(shape) * np.log(2 * np.pi * var),
                dtype=torch.float64,
            ),
            persistent=False,
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -(0.5 / self._var) * torchutils.sum_except_batch(
            inputs**2, num_batch_dims=1
        )
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.normal(
                0,
                self._std,
                size=(num_samples, *self._shape),
                device=self._log_z.device,
            )
        else:
            raise NotImplementedError

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            raise NotImplementedError


class ResampledGaussian(BaseResampledGaussian):
    """Wrapper for ResampledGaussian.

    Adds methods needed in nessai.
    """

    end_iteration = BaseResampledGaussian.estimate_normalisation_constant
    """Function to be called at the end of an iteration.

    For LARS this updates the estimate of the normalisation constant
    independently of the other parameters in the flow.
    """

    def finalise(self, n_samples: int = 10_000, n_batches: int = 10) -> None:
        """Finalise the estimate of the normalisation constant."""
        self.estimate_normalisation_constant(
            n_samples=n_samples, n_batches=n_batches
        )
