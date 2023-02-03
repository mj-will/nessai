# -*- coding: utf-8 -*-
"""
Base reparameterisation
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Reparameterisation:
    """
    Base object for reparameterisations.

    Parameters
    ----------
    parameters : str or list
        Name of parameters to reparameterise.
    prior_bounds : list, dict or None
        Prior bounds for the parameter(s).
    """

    _update_bounds = False
    has_prior = False
    has_prime_prior = False
    requires_prime_prior = False
    requires_bounded_prior = False
    prior_bounds = None
    prime_prior_bounds = None

    def __init__(self, parameters=None, prior_bounds=None):
        if not isinstance(parameters, (str, list)):
            raise TypeError("Parameters must be a str or list.")

        self.parameters = (
            [parameters] if isinstance(parameters, str) else parameters.copy()
        )

        if isinstance(prior_bounds, (list, tuple, np.ndarray)):
            if len(prior_bounds) == 2:
                prior_bounds = {self.parameters[0]: np.asarray(prior_bounds)}
            else:
                raise RuntimeError("Prior bounds got a list of len > 2")
        elif prior_bounds is None:
            if self.requires_bounded_prior:
                raise RuntimeError(
                    f"Reparameterisation {self.name} requires prior bounds!"
                )
            else:
                self.prior_bounds = None
        elif not isinstance(prior_bounds, dict):
            raise TypeError(
                "Prior bounds must be a dict, tuple, list or numpy array"
                " of len 2 or None."
            )

        if prior_bounds is not None:
            if set(self.parameters) - set(prior_bounds.keys()):
                raise RuntimeError(
                    "Mismatch between parameters and prior bounds: "
                    f"{set(self.parameters)}, {set(prior_bounds.keys())}"
                )
            self.prior_bounds = {
                p: np.asarray(b) for p, b in prior_bounds.items()
            }
        else:
            logger.debug(f"No prior bounds for {self.name}")

        if self.requires_bounded_prior:
            is_finite = np.isfinite(
                [pb for pb in self.prior_bounds.values()]
            ).all()
            if not is_finite:
                raise RuntimeError(
                    f"Reparameterisation {self.name} requires finite prior "
                    f"bounds. Received: {self.prior_bounds}"
                )

        self.prime_parameters = [p + "_prime" for p in self.parameters]
        self.requires = []
        logger.debug(f"Initialised reparameterisation: {self.name}")

    @property
    def name(self):
        """Unique name of the reparameterisations"""
        return (
            self.__class__.__name__.lower() + "_" + "_".join(self.parameters)
        )

    def reparameterise(self, x, x_prime, log_j):
        """
        Apply the reparameterisation to convert from x-space to x'-space.

        Parameters
        ----------
        x : structured array
            Array of inputs
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        raise NotImplementedError

    def inverse_reparameterise(self, x, x_prime, log_j):
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

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        raise NotImplementedError

    def update(self, x):
        """Update the reparameterisation given some points.

        Does nothing by default.
        """
        pass
