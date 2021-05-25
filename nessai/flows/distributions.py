# -*- coding: utf-8 -*-
"""
Distributions to use as the 'base distribution' for normalising flows.
"""

from nflows.distributions import Distribution
from nflows.distributions.uniform import BoxUniform as BaseBoxUniform
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


class BoxUniform(BaseBoxUniform):
    """Wrapper to `nflows.distributions.uniform.BoxUniform`"""

    def sample(self, n=1):
        """Sample from the box uniform"""
        return super().sample((n,))

    def sample_and_log_prob(self, n=1):
        """Sample from the distribution and compute the log prob"""
        x = self.sample(n)
        log_prob = self.log_prob(x)
        return x, log_prob
