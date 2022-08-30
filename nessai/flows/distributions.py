# -*- coding: utf-8 -*-
"""
Distributions to use as the 'base distribution' for normalising flows.
"""
import logging
from typing import Callable

from nflows.distributions import Distribution
from nflows.utils import torchutils
import numpy as np
import torch
from torch import nn

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
        Variance of the distribution.
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


class ResampledGaussian(Distribution):
    """Gaussian distribution that includes LARS.

    For details see: https://arxiv.org/abs/2110.15828

    Based on the implementation here: \
        https://github.com/VincentStimper/resampled-base-flows

    Does not support conditional inputs.

    Parameters
    ----------
    shape
        Shape of the distribution
    acceptance_fn
        Function that computes the acceptance. Typically a neural network.
    eps
        Decay parameter for the exponential moving average used to update
        the estimate of Z.
    truncation
        Maximum number of rejection steps. Called T in the original paper.
    trainable
        Boolean to indicate if the mean and standard deviation of the
        distribution are learnable parameters.
    """
    def __init__(
        self, shape: tuple,
        acceptance_fn: Callable,
        eps: float = 0.05,
        truncation: int = 100,
        trainable: bool = False
    ) -> None:
        super().__init__()
        self._shape = torch.Size(shape)
        self.truncation = truncation
        self.acceptance_fn = acceptance_fn
        self.eps = eps

        self.register_buffer("norm", torch.tensor(-1.0))
        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64
            ),
        )
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *shape))
            self.register_buffer("log_scale", torch.zeros(1, *shape))

    def _log_prob_gaussian(self, norm_inputs: torch.tensor) -> torch.tensor:
        """Base Gaussian log probability"""
        log_prob = (
            -0.5
            * torchutils.sum_except_batch(norm_inputs ** 2, num_batch_dims=1)
            - torchutils.sum_except_batch(self.log_scale, num_batch_dims=1)
            - self._log_z
        )
        return log_prob

    def _log_prob(
        self, inputs: torch.tensor, context: torch.tensor = None
    ) -> torch.tensor:
        """Log probability"""
        if context is not None:
            raise ValueError("Conditional inputs not supported")

        norm_inputs = (inputs - self.loc) / self.log_scale.exp()
        log_p_gaussian = self._log_prob_gaussian(norm_inputs)
        acc = self.acceptance_fn(norm_inputs)

        if self.training or self.norm < 0.0:
            eps_ = torch.randn_like(inputs)
            norm_batch = torch.mean(self.acceptance_fn(eps_))
            if self.norm < 0.0:
                self.norm = norm_batch.detach()
            else:
                # Update the normalisation estimate
                # eps defines the weight between the current estimate
                # and the new estimated value
                self.norm = (
                    1 - self.eps
                ) * self.norm + self.eps * norm_batch.detach()
            # Why this?
            norm = norm_batch - norm_batch.detach() + self.norm
        else:
            norm = self.norm

        alpha = (1 - norm) ** (self.truncation - 1)
        return (
            torch.log((1 - alpha) * acc[:, 0] / norm + alpha) + log_p_gaussian
        )

    def _sample(
        self, num_samples: int, context: torch.tensor = None
    ) -> torch.tensor:
        if context is not None:
            raise ValueError("Conditional inputs not supported")

        device = self._log_z.device
        samples = torch.zeros(num_samples, *self._shape, device=device)

        t = 0
        s = 0
        n = 0
        norm_sum = 0

        for _ in range(self.truncation):
            samples_ = torch.randn(num_samples, *self._shape, device=device)
            acc = self.acceptance_fn(samples_)
            if self.training or self.norm < 0:
                norm_sum = norm_sum + acc.sum().detach()
                n += num_samples

            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or (t == (self.truncation - 1)):
                    samples[s, :] = samples_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break

        samples = self.loc + self.log_scale.exp() * samples
        return samples

    def estimate_normalisation_constant(
        self, n_samples: int = 1000, n_batches: int = 1
    ) -> None:
        """Estimate the normalisation constant via Monte Carlo sampling.

        Should be called once the training is complete.

        Parameters
        ----------
        n_samples
            Number of samples to draw in each batch.
        n_batches
            Number of batches to use.
        """
        with torch.no_grad():
            self.norm = self.norm * 0.0
            # Get dtype and device
            dtype = self.norm.dtype
            device = self.norm.device
            for _ in range(n_batches):
                eps = torch.randn(
                    n_samples, *self._shape,
                    dtype=dtype, device=device
                )
                acc_ = self.acceptance_fn(eps)
                norm_batch = torch.mean(acc_)
                self.norm = self.norm + norm_batch / n_batches

    end_iteration = estimate_normalisation_constant
    """Function to be called at the end of an iteration.

    For LARS this updates the estimate of the normalisation constant
    independently of the other parameters in the flow.
    """

    def finalise(self, n_samples: int = 10_000, n_batches: int = 10) -> None:
        """Finalise the estimate of the normalisation constant."""
        logger.debug('Computing final estimate of the normalisation constant')
        self.estimate_normalisation_constant(
            n_samples=n_samples, n_batches=n_batches
        )


class DiagonalNormal(Distribution):
    """A diagonal multivariate Normal with trainable parameters.

    Based on the implementation in nflows.

    Parameters
    ----------
    shape : tuple
        Shape on the input variables.
    """

    def __init__(self, shape):
        super().__init__()
        assert len(shape) == 1
        self._shape = torch.Size(shape)
        self.mean_ = nn.Parameter(torch.zeros(shape).reshape(1, -1))
        self.log_std_ = nn.Parameter(torch.zeros(shape).reshape(1, -1))
        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64
            ),
            persistent=False,
            )

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        # Compute parameters.
        means = self.mean_
        log_stds = self.log_std_

        # Compute log prob.
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(
            norm_inputs ** 2, num_batch_dims=1
        )
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        if context is not None:
            raise NotImplementedError()
        # This is not optimal and could be improved significantly.
        with torch.no_grad():
            dist = torch.distributions.MultivariateNormal(
                self.mean_.data.flatten(),
                np.exp(2 * self.log_std_.flatten()) * torch.eye(self._shape[0])
            )
            out = dist.sample((num_samples,))
        return out

    def _mean(self, context):
        return self.mean
