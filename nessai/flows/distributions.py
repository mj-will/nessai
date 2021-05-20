# -*- coding: utf-8 -*-
"""
Distributions to use as the 'base distribution' for normalising flows.
"""
import logging
import math

from nflows.distributions import Distribution
from nflows.distributions.uniform import BoxUniform as BaseBoxUniform
from nflows.utils import torchutils
import numpy as np
from scipy.special import gamma
import torch


logger = logging.getLogger(__name__)


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


class SphericalTruncatedNormal(Distribution):
    """
    A Gaussian distribution truncated to lie within an n-ball of radius r.

    Based on the implentation by toshas: \
        https://github.com/toshas/torch_truncnorm

    Parameters
    ----------
    shape : int or tuple
        Shape of the inputs
    r : float
        Radius of the n-ball
    """
    def __init__(self, shape, r):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = torch.Size(shape)

        logger.debug(f'Truncated normal with r={r:.3}')

        self.register_buffer('r', torch.tensor(r, dtype=torch.float64))

        self.register_buffer(
            "_log_z",
            (
                np.prod(self._shape)
                * (-0.5 * np.log(2 * np.pi)
                    - torch.log(self._phi(self.r) - self._phi(-self.r)))
            ),
            persistent=False)

    @staticmethod
    def _phi(x):
        return 0.5 * (1.0 + (x / math.sqrt(2.0)).erf())

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        ib = self.r.gt(torch.linalg.norm(inputs, dim=1)).type_as(inputs)
        return torch.log(ib) + self._log_z \
            - 0.5 * torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)

    def _sample(self, num_samples, context):
        raise RuntimeError


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


class UniformNBall(Distribution):
    """
    Uniform distribution in an n-ball
    """
    def __init__(self, shape, radius=1.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = torch.Size(shape)

        self.register_buffer(
            '_radius',
            torch.tensor(float(radius))
        )

        self.register_buffer(
            '_radius_sq',
            torch.tensor(float(radius) ** 2)
        )
        self.register_buffer(
            '_log_vol',
            torch.tensor(
                shape[-1] / 2 * np.log(np.pi) +
                shape[-1] * np.log(radius) -
                np.log(gamma(shape[-1] / 2 + 1))
            )
        )

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        ib = self._radius_sq.gt(
            torch.pow(inputs, 2.0).sum(dim=-1)).type_as(inputs)
        return torch.log(ib) - self._log_vol

    def _sample(self, n, context):
        if context is None:
            pass
            x = torch.randn(n, *self._shape, device=self._log_vol.device)
            r = torch.rand(
                n, 1, *self._shape[:-1], device=self._log_vol.device)
            s = r ** (1 / self._shape[-1]) * x / torch.linalg.norm(x)
            return self._radius * s
        else:
            raise RuntimeError

    def sample_and_log_prob(self, n=1):
        """Sample from the distribution and compute the log prob"""
        x = self.sample(n)
        log_prob = self.log_prob(x)
        return x, log_prob
