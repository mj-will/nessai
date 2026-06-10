# -*- coding: utf-8 -*-
"""
Base reparameterisation
"""

from __future__ import annotations

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
    prime_parameters : str or list, optional
        Name of the parameters in the prime space. If None, will be set to
        the same as `parameters` with '_prime' appended.
    auxiliary_parameters : str or list, optional
        Name of any auxiliary parameters that are made available in x-space after
        the reparameterisation. These parameters are not required for the forward
        pass but may be used in the inverse pass. Defaults to None.
    prior_bounds : list, dict or None
        Prior bounds for the parameter(s).
    rng: np.random.Generator, optional
        Random number generator to use for any random operations in the
        reparameterisation. If None, a new default_rng will be used.
    requires : str or list, optional
        Name of any parameters that are required for the reparameterisation.
    prime_requires : str or list, optional
        Name of any parameters in the prime space that are required for the
        reparameterisation.
    inverse_requires : str or list, optional
        Name of any parameters that are required for the inverse
        reparameterisation.
    """

    _update = False
    has_prior = False
    requires_bounded_prior = False
    prior_bounds = None
    one_to_one = True

    def __init__(
        self,
        parameters=None,
        prime_parameters=None,
        auxiliary_parameters=None,
        prior_bounds=None,
        rng=None,
        requires=None,
        prime_requires=None,
        inverse_requires=None,
    ):
        if rng is None:
            logger.debug("No rng specified, using the default rng.")
            rng = np.random.default_rng()
        self.rng = rng
        if not isinstance(parameters, (str, list)):
            raise TypeError("Parameters must be a str or list.")

        self.parameters = self._format_parameters(parameters)

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
            missing_bounds = set(self.parameters) - set(prior_bounds.keys())
            if missing_bounds and self.requires_bounded_prior:
                raise RuntimeError(
                    "Mismatch between parameters and prior bounds: "
                    f"{set(self.parameters)}, {set(prior_bounds.keys())}"
                )
            self.prior_bounds = {
                p: np.asarray(b) for p, b in prior_bounds.items()
            }
            if missing_bounds:
                logger.debug(
                    "Missing prior bounds for parameters %s in %s",
                    sorted(missing_bounds),
                    self.name,
                )
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

        self.prime_parameters = self._format_parameters(prime_parameters) or [
            f"{p}_prime" for p in self.parameters
        ]
        self.auxiliary_parameters = self._format_parameters(
            auxiliary_parameters
        )
        self.requires = self._format_parameters(requires)
        self.prime_requires = self._format_parameters(prime_requires)
        self.inverse_requires = self._format_parameters(inverse_requires)
        logger.debug(f"Initialised reparameterisation: {self.name}")

    @staticmethod
    def _format_parameters(parameters: str | list[str] | None) -> list[str]:
        """Format the parameters to be a list of strings."""
        if isinstance(parameters, str):
            return [parameters]
        elif isinstance(parameters, list):
            return parameters.copy()
        elif parameters is None:
            return []
        else:
            raise TypeError(
                "Parameters must be a string or a list of strings."
            )

    @property
    def output_parameters(self):
        """All x-space parameters made available after this reparameterisation."""
        return self.parameters + self.auxiliary_parameters

    @property
    def input_parameters(self):
        """All x-space parameters required for the forward pass."""
        # self.parameters + self.requires may contain duplicates
        # we want to preserve the order but remove duplicates
        return list(dict.fromkeys(self.parameters + self.requires))

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

    def reset(self):
        """Reset the reparameterisation.

        Does nothing by default.
        """
        pass
