# -*- coding: utf-8 -*-
"""
Null reparameterisation.
"""

import logging

from .base import Reparameterisation

logger = logging.getLogger(__name__)


class IdentityReparameterisation(Reparameterisation):
    """Reparameterisation that does not modify the parameters.

    Parameters
    ----------
    parameters : list or str
        Parameters for which the reparameterisation will be used.
    prior_bounds : list, dict or None
        Prior bounds for parameters. Unused for this reparameterisation.
    """

    def __init__(
        self,
        input_parameters=None,
        output_parameters=None,
        persistent_parameters=None,
        auxiliary_parameters=None,
        prior_bounds=None,
        rng=None,
        parameters=None,
    ):
        super().__init__(
            input_parameters=input_parameters,
            output_parameters=output_parameters,
            persistent_parameters=persistent_parameters,
            auxiliary_parameters=auxiliary_parameters,
            prior_bounds=prior_bounds,
            rng=rng,
            parameters=parameters,
        )
        self.output_parameters = self.input_parameters
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
        for parameter, output_parameter in zip(
            self.parameters, self.output_parameters
        ):
            x_prime[output_parameter] = self.get_value(parameter, x, x_prime)
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
        for parameter, output_parameter in zip(
            self.parameters, self.output_parameters
        ):
            x, x_prime = self._set_value(
                parameter, x_prime[output_parameter], x, x_prime
            )
        return x, x_prime, log_j


NullReparameterisation = IdentityReparameterisation
"""Alias for IdentityReparameterisation."""
