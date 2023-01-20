# -*- coding: utf-8 -*-
"""
Null reparameterisation.
"""
import logging

from .base import Reparameterisation

logger = logging.getLogger(__name__)


class NullReparameterisation(Reparameterisation):
    """Reparameterisation that does not modify the parameters.

    Parameters
    ----------
    parameters : list or str
        Parameters for which the reparameterisation will be used.
    prior_bounds : list, dict or None
        Prior bounds for parameters. Unused for this reparameterisation.
    """

    def __init__(self, parameters=None, prior_bounds=None):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)
        self.prime_parameters = self.parameters
        logger.debug(f"Initialised reparameterisation: {self.name}")

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space
        to x'-space

        Parameters
        ----------
        x : structured array
            Array
        x_prime : structured array
            Array to be update
        log_j : Log jacobian to be updated
        """
        x_prime[self.prime_parameters] = x[self.parameters]
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space
        to x'-space

        Parameters
        ----------
        x : structured array
            Array
        x_prime : structured array
            Array to be update
        log_j : Log jacobian to be updated
        """
        x[self.parameters] = x_prime[self.prime_parameters]
        return x, x_prime, log_j
