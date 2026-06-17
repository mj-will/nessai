# -*- coding: utf-8 -*-
"""
Combined reparameterisation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..livepoint import empty_structured_array
from ..utils.sorting import sort_reparameterisations

if TYPE_CHECKING:
    from . import Reparameterisation

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

    def __init__(
        self,
        reparameterisations: list["Reparameterisation"] = None,
        reverse_order: bool = False,
        initial_parameters: list[str] | None = None,
    ):
        super().__init__()
        self.reparameterisations = {}
        self.parameters = []
        self.prime_parameters = []
        self.order = []
        self.reverse_order = reverse_order
        self.initial_parameters = (
            initial_parameters.copy() if initial_parameters is not None else []
        )
        if reparameterisations is not None:
            self.add_reparameterisations(reparameterisations)

    @property
    def one_to_one(self):
        return all(r.one_to_one for r in self.values())

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
        available_parameters = self.initial_parameters + self.parameters
        available_prime_parameters = self.prime_parameters
        missing_parameters = reparameterisation.resolve_forward_input_spaces(
            available_parameters, available_prime_parameters
        )
        if missing_parameters:
            raise RuntimeError(
                f"Could not add {reparameterisation}, missing requirement(s): "
                f"{missing_parameters}. Current: {available_parameters}. "
                f"Current prime parameters: {available_prime_parameters}."
            )

        self[reparameterisation.name] = reparameterisation
        self.order = list(self.keys())
        self.parameters += [
            p
            for p in reparameterisation.x_output_parameters
            if p not in self.parameters
        ]
        self.prime_parameters += [
            p
            for p in reparameterisation.output_parameters
            if p not in self.prime_parameters
        ]

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
        existing_parameters = self.initial_parameters + self.parameters
        existing_prime_parameters = self.prime_parameters
        logger.debug(f"Existing parameters: {existing_parameters}")
        reparameterisations = sort_reparameterisations(
            reparameterisations,
            existing_parameters=existing_parameters,
            existing_prime_parameters=existing_prime_parameters,
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
        prime_parameters = self.prime_parameters.copy()
        for key in self.from_prime_order:
            missing_parameters = self[key].resolve_inverse_input_spaces(
                parameters, prime_parameters
            )
            if missing_parameters:
                raise RuntimeError(
                    "Order of reparameterisations is invalid (x' -> x)"
                    f"{self[key].name} requires "
                    f"{self[key].inverse_input_parameters} but with "
                    f"the current order only the parameters {parameters} "
                    "would be available."
                )
            parameters += [
                p for p in self[key].x_output_parameters if p not in parameters
            ]

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
        x_current = x.copy()
        x_prime = empty_structured_array(x.size, names=self.prime_parameters)
        log_j = np.zeros(x.size)

        for key in self.to_prime_order:
            reparameterisation = self[key]
            if hasattr(reparameterisation, "update_bounds"):
                logger.debug(f"Updating bounds for: {reparameterisation.name}")
                reparameterisation.update_bounds(x_current, x_prime=x_prime)
            x_current, x_prime, log_j = reparameterisation.reparameterise(
                x_current, x_prime, log_j
            )

    def reset_inversion(self):
        """
        Reset edges for boundary inversion
        """
        for r in self.values():
            if hasattr(r, "reset_inversion"):
                r.reset_inversion()

    def update(self, x):
        """Update the reparameterisations given a set of points."""
        x_current = x.copy()
        x_prime = empty_structured_array(x.size, names=self.prime_parameters)
        log_j = np.zeros(x.size)

        for key in self.to_prime_order:
            reparameterisation = self[key]
            reparameterisation.update(x_current, x_prime=x_prime)
            x_current, x_prime, log_j = reparameterisation.reparameterise(
                x_current, x_prime, log_j
            )

    def reset(self):
        """Reset the reparameterisations"""
        for r in self.values():
            r.reset()

    def log_prior(self, x):
        """
        Compute any additional priors for auxiliary parameters
        """
        log_p = np.zeros(x.size)
        for r in self.values():
            if r.has_prior:
                log_p += r.log_prior(x)
        return log_p
