# -*- coding: utf-8 -*-
"""
Distributions to use as the 'base distribution' for normalising flows.
"""
from typing import Union

from nflows.distributions import Distribution
from nflows.utils import torchutils
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
        Variance of the distrinution.
    """

    def __init__(self, shape, var=1):
        super().__init__()
        self._shape = torch.Size(shape)
        self._var = var

        self.register_buffer(
            "_log_z",
            torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi * var),
                         dtype=torch.float64),
            persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -(0.5 / self._var) * \
            torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.normal(0, self._var,
                                size=(num_samples, *self._shape),
                                device=self._log_z.device)
        else:
            raise NotImplementedError

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            raise NotImplementedError


class BoxUniform(Distribution):
    def __init__(
        self,
        low: Union[torch.Tensor, float],
        high: Union[torch.Tensor, float]
    ):
        """Multidimensional uniform distribution defined on a box.

        Based on this implementation: \
            https://github.com/bayesiains/nflows/pull/17 but with fixes for
            CUDA support.

        Parameters
        -----------
        low : Tensor or float
            Lower range (inclusive).
        high : Tensor or float
            Upper range (exclusive).
        """
        super().__init__()
        if low.shape != high.shape:
            raise ValueError(
                "low and high are not of the same size"
            )

        if not (low < high).byte().all():
            raise ValueError(
                "low has elements that are higher than high"
            )

        self._shape = low.shape
        self.register_buffer('low', low)
        self.register_buffer('high', high)
        self.register_buffer(
            '_log_prob_value',
            -torch.sum(torch.log(high - low))
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        lb = self.low.le(inputs).type_as(self.low).prod(-1)
        ub = self.high.gt(inputs).type_as(self.low).prod(-1)
        return torch.log(lb.mul(ub)) - self._log_prob_value

    def _sample(self, num_samples, context):
        context_size = 1 if context is None else context.shape[0]
        low_expanded = \
            self.low.expand(context_size * num_samples, *self._shape)
        high_expanded = \
            self.high.expand(context_size * num_samples, *self._shape)
        samples = \
            low_expanded + \
            torch.rand(
                context_size * num_samples,
                *self._shape,
                device=self.low.device
            ) * \
            (high_expanded - low_expanded)
        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(
                samples, [context_size, num_samples]
            )
