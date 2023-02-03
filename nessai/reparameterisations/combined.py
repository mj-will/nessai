# -*- coding: utf-8 -*-
"""
Combined reparameterisation.
"""
import logging

import numpy as np

from ..utils.sorting import sort_reparameterisations

logger = logging.getLogger(__name__)


class CombinedReparameterisation(dict):
    """Class to handle multiple reparameterisations

    Parameters
    ----------
    reparameterisations : list, optional
        List of reparameterisations to add to the combined reparameterisations.
        Further reparameterisations can be added using
        `add_reparameterisations`.
    reverse_order : bool
        If True the order of the reparameterisations will be reversed compared
        to the default ordering.
    """

    def __init__(self, reparameterisations=None, reverse_order=False):
        super().__init__()
        self.reparameterisations = {}
        self.parameters = []
        self.prime_parameters = []
        self.requires = []
        self.order = []
        self.reverse_order = reverse_order
        if reparameterisations is not None:
            self.add_reparameterisations(reparameterisations)

    @property
    def has_prime_prior(self):
        """Boolean to check if prime prior can be enabled"""
        return all(r.has_prime_prior for r in self.values())

    @property
    def requires_prime_prior(self):
        """Boolean to check if any of the priors require the prime space"""
        return any(r.requires_prime_prior for r in self.values())

    @property
    def to_prime_order(self):
        """Order when converting to the prime space"""
        if self.reverse_order:
            return reversed(self.order)
        else:
            return self.order

    @property
    def from_prime_order(self):
        """Order when converting from the prime space"""
        if self.reverse_order:
            return self.order
        else:
            return reversed(self.order)

    def _add_reparameterisation(self, reparameterisation):
        requires = reparameterisation.requires
        if requires and (
            any([req not in self.parameters for req in requires])
            and any([req not in self.prime_parameters for req in requires])
        ):
            raise RuntimeError(
                f"Could not add {reparameterisation}, missing requirement(s): "
                f"{reparameterisation.requires}. Current: {self.parameters}."
            )

        self[reparameterisation.name] = reparameterisation
        self.order = list(self.keys())
        self.parameters += reparameterisation.parameters
        self.prime_parameters += reparameterisation.prime_parameters
        self.requires += reparameterisation.requires

    def add_reparameterisation(self, reparameterisation):
        """Add a reparameterisation"""
        self.add_reparameterisations(reparameterisation)

    def add_reparameterisations(self, reparameterisations):
        """Add multiple reparameterisations

        Parameters
        ----------
        reparameterisations : list of :`obj`:Reparameterisation
            List of reparameterisations to add.
        """
        if not isinstance(reparameterisations, list):
            reparameterisations = [reparameterisations]

        logger.debug("Sorting reparameterisations")
        logger.debug(f"Existing parameters: {self.parameters}")
        reparameterisations = sort_reparameterisations(
            reparameterisations,
            existing_parameters=self.parameters,
        )

        for r in reparameterisations:
            self._add_reparameterisation(r)

    def check_order(self):
        """Check the order of the reparameterisations is valid.

        Raises
        ------
        RuntimeError
            Raised if the order is invalid.
        """
        parameters = []
        for key in self.from_prime_order:
            if not all([r in parameters for r in self[key].requires]):
                raise RuntimeError(
                    "Order of reparameterisations is invalid (x' -> x)"
                    f"{self[key].name} requires {self[key].requires} but with "
                    f"the current order only the parameters {parameters} "
                    "would be available."
                )
            parameters += self[key].parameters

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
        log_j : array_like
            Log jacobian to be updated
        """
        for key in self.to_prime_order:
            x, x_prime, log_j = self[key].reparameterise(
                x, x_prime, log_j, **kwargs
            )
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
        log_j : array_like
            Log jacobian to be updated
        """
        for key in self.from_prime_order:
            x, x_prime, log_j = self[key].inverse_reparameterise(
                x, x_prime, log_j, **kwargs
            )
        return x, x_prime, log_j

    def update_bounds(self, x):
        """
        Update the bounds used for the reparameterisation
        """
        for r in self.values():
            if hasattr(r, "update_bounds"):
                logger.debug(f"Updating bounds for: {r.name}")
                r.update_bounds(x)

    def reset_inversion(self):
        """
        Reset edges for boundary inversion
        """
        for r in self.values():
            if hasattr(r, "reset_inversion"):
                r.reset_inversion()

    def update(self, x):
        """Update the reparameterisations given a set of points."""
        for r in self.values():
            r.update(x)

    def log_prior(self, x):
        """
        Compute any additional priors for auxiliary parameters
        """
        log_p = np.zeros(x.size)
        for r in self.values():
            if r.has_prior:
                log_p += r.log_prior(x)
        return log_p

    def x_prime_log_prior(self, x_prime):
        """
        Compute the prior in the prime space
        """
        log_p = np.zeros(x_prime.size)
        for r in self.values():
            log_p += r.x_prime_log_prior(x_prime)
        return log_p
