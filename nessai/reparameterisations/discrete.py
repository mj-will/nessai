"""Reparameterisations for discrete variables"""
import numpy as np

from .base import Reparameterisation
from ..utils.rescaling import (
    rescale_zero_to_one,
    inverse_rescale_zero_to_one,
    sigmoid,
    logit,
)


class Dequantise(Reparameterisation):
    """Dequantise discrete variables.

    Output is defined on [0, 1) unless the logit is applied.

    Parameters
    ----------
    parameters : list
        List of parameter names
    prior_bounds : dict
        Dictionary of prior bounds
    include_logit : bool
        If true, the a logit is applied after dequantising.
    eps : Optional[float]
        Epsilon value used for the logit.
    """

    requires_bounded_prior = True

    def __init__(
        self, parameters=None, prior_bounds=None, include_logit=False, eps=None
    ):
        super().__init__(parameters, prior_bounds)
        self.prime_parameters = [p + "_dequant" for p in self.parameters]

        self.include_logit = include_logit
        self.eps = eps
        if self.prior_bounds is None:
            raise RuntimeError("Must specify priors for dequantise")

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Reparameterise the discrete parameters"""
        for p, pp in zip(self.parameters, self.prime_parameters):
            x_prime[pp], lj = rescale_zero_to_one(
                x[p] + np.random.rand(x.size),
                xmin=self.prior_bounds[p][0],
                xmax=self.prior_bounds[p][1] + 1,
            )
            log_j += lj

            if self.include_logit:
                x_prime[pp], lj = logit(x_prime[pp], eps=self.eps)
                log_j += lj

        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Quantise the prime parameters"""
        for p, pp in zip(self.parameters, self.prime_parameters):

            if self.include_logit:
                x_tmp, lj = sigmoid(x_prime[pp])
                log_j += lj
            else:
                x_tmp = x_prime[pp]

            x_tmp, lj = inverse_rescale_zero_to_one(
                x_tmp,
                xmin=self.prior_bounds[p][0],
                xmax=self.prior_bounds[p][1] + 1,
            )
            x[p] = np.floor(x_tmp)
            log_j += lj
        return x, x_prime, log_j
