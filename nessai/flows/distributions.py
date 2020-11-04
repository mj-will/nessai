
from nflows.distributions import Distribution
from nflows.utils import torchutils
import numpy as np
import torch


class MultivariateNormal(Distribution):
    """A multivariate Normal with zero mean and specified covariance."""

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
