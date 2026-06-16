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
    input_parameters : str or list
        Names of the parameters required in the forward direction.
    output_parameters : str or list, optional
        Names of the parameters produced in the prime space. If None, will be
        set to the same as `input_parameters` with '_prime' appended.
    persistent_parameters : str or list, optional
        Subset of `input_parameters` that should remain exposed in the
        flow-facing parameter set after this reparameterisation.
    auxiliary_parameters : str or list, optional
        Name of any auxiliary parameters that are made available in x-space after
        the reparameterisation. These parameters are not required for the forward
        pass but may be used in the inverse pass. Defaults to None.
    prior_bounds : list, dict or None
        Prior bounds for the parameter(s).
    rng: np.random.Generator, optional
        Random number generator to use for any random operations in the
        reparameterisation. If None, a new default_rng will be used.
    inverse_input_parameters : str or list, optional
        Name of any parameters that are required for the inverse
        reparameterisation.
    parameters : str or list, optional
        Alias for `input_parameters`.
    """

    _update = False
    has_prior = False
    requires_bounded_prior = False
    prior_bounds = None
    one_to_one = True

    def __init__(
        self,
        input_parameters=None,
        output_parameters=None,
        persistent_parameters=None,
        auxiliary_parameters=None,
        prior_bounds=None,
        rng=None,
        inverse_input_parameters=None,
        parameters=None,
    ):
        if rng is None:
            logger.debug("No rng specified, using the default rng.")
            rng = np.random.default_rng()
        self.rng = rng
        if parameters is not None and input_parameters is not None:
            if self._format_parameters(parameters) != self._format_parameters(
                input_parameters
            ):
                raise RuntimeError(
                    "Received conflicting values for `parameters` and "
                    "`input_parameters`."
                )
        if input_parameters is None:
            input_parameters = parameters
        if not isinstance(input_parameters, (str, list)):
            raise TypeError("Parameters must be a str or list.")

        self.input_parameters = self._format_parameters(input_parameters)

        if isinstance(prior_bounds, (list, tuple, np.ndarray)):
            if len(prior_bounds) == 2:
                prior_bounds = {
                    self.input_parameters[0]: np.asarray(prior_bounds)
                }
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
            missing_bounds = set(self.input_parameters) - set(
                prior_bounds.keys()
            )
            if missing_bounds and self.requires_bounded_prior:
                raise RuntimeError(
                    "Mismatch between parameters and prior bounds: "
                    f"{set(self.input_parameters)}, {set(prior_bounds.keys())}"
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

        self.output_parameters = self._format_parameters(
            output_parameters
        ) or [f"{p}_prime" for p in self.input_parameters]
        self.persistent_parameters = self._format_parameters(
            persistent_parameters
        )
        if not set(self.persistent_parameters).issubset(self.input_parameters):
            raise RuntimeError(
                "Persistent parameters must be a subset of the input "
                f"parameters. Received {self.persistent_parameters} for "
                f"{self.input_parameters}."
            )
        self.auxiliary_parameters = self._format_parameters(
            auxiliary_parameters
        )
        self.inverse_input_parameters = self._format_parameters(
            inverse_input_parameters
        )
        self._x_input_parameters = []
        self._x_prime_input_parameters = []
        self._x_persistent_parameters = []
        self._x_prime_persistent_parameters = []
        self._x_inverse_input_parameters = []
        self._x_prime_inverse_input_parameters = []
        self._resolved_forward_inputs = False
        self._resolved_inverse_inputs = False
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
    def x_input_parameters(self):
        """Resolved x-space forward inputs."""
        if self._resolved_forward_inputs:
            return self._x_input_parameters.copy()
        return self.input_parameters.copy()

    @property
    def x_prime_input_parameters(self):
        """Resolved x'-space forward inputs."""
        return self._x_prime_input_parameters.copy()

    @property
    def prime_input_parameters(self):
        """Compatibility alias for `x_prime_input_parameters`."""
        return self.x_prime_input_parameters

    @property
    def x_output_parameters(self):
        """All x-space parameters available after this reparameterisation."""
        return list(
            dict.fromkeys(self.x_input_parameters + self.auxiliary_parameters)
        )

    @property
    def x_persistent_parameters(self):
        """Resolved persistent x-space inputs."""
        return self._x_persistent_parameters.copy()

    @property
    def x_prime_persistent_parameters(self):
        """Resolved persistent x'-space inputs."""
        return self._x_prime_persistent_parameters.copy()

    @property
    def x_inverse_input_parameters(self):
        """Resolved x-space inverse inputs."""
        return self._x_inverse_input_parameters.copy()

    @property
    def x_prime_inverse_input_parameters(self):
        """Resolved x'-space inverse inputs."""
        return self._x_prime_inverse_input_parameters.copy()

    @property
    def parameters(self):
        """Compatibility alias for `input_parameters`."""
        return self.input_parameters

    @parameters.setter
    def parameters(self, value):
        self.input_parameters = self._format_parameters(value)
        self._resolved_forward_inputs = False
        self._resolved_inverse_inputs = False

    @property
    def name(self):
        """Unique name of the reparameterisations"""
        return (
            self.__class__.__name__.lower()
            + "_"
            + "_".join(self.input_parameters)
        )

    def resolve_forward_input_spaces(
        self, available_parameters, available_prime_parameters
    ):
        """Resolve forward inputs against x and prime namespaces."""
        x_inputs = []
        prime_inputs = []
        missing = []
        for parameter in self.input_parameters:
            if parameter in available_parameters:
                x_inputs.append(parameter)
            elif parameter in available_prime_parameters:
                prime_inputs.append(parameter)
            else:
                missing.append(parameter)

        x_persistent = [
            parameter
            for parameter in self.persistent_parameters
            if parameter in x_inputs
        ]

        x_prime_persistent = [
            parameter
            for parameter in self.persistent_parameters
            if parameter in prime_inputs
        ]

        self._x_input_parameters = x_inputs
        self._x_prime_input_parameters = prime_inputs
        self._x_persistent_parameters = x_persistent
        self._x_prime_persistent_parameters = x_prime_persistent
        self._resolved_forward_inputs = True
        return missing

    def resolve_inverse_input_spaces(
        self, available_parameters, available_prime_parameters
    ):
        """Resolve inverse inputs against x and prime namespaces."""
        x_inputs = []
        x_prime_inputs = []
        missing = []
        for parameter in self.inverse_input_parameters:
            if parameter in available_parameters:
                x_inputs.append(parameter)
            elif parameter in available_prime_parameters:
                x_prime_inputs.append(parameter)
            else:
                missing.append(parameter)

        self._x_inverse_input_parameters = x_inputs
        self._x_prime_inverse_input_parameters = x_prime_inputs
        self._resolved_inverse_inputs = True
        return missing

    def _get_value(self, parameter, x, x_prime=None):
        """Get the current value for an input parameter.

        Returns the value from x or x_prime depending on where the parameter is
        defined.
        """
        x_prime_inputs = getattr(self, "_x_prime_input_parameters", [])
        if not isinstance(x_prime_inputs, (list, tuple, set)):
            x_prime_inputs = []
        if parameter in x_prime_inputs:
            if x_prime is None:
                raise RuntimeError(
                    f"Prime-space input `{parameter}` requested for "
                    f"{self.name} but no x_prime array was provided."
                )
            return x_prime[parameter]
        return x[parameter]

    def _set_value(self, parameter, value, x, x_prime=None):
        """Set the reconstructed value for an input parameter.

        Sets the value in x or x_prime depending on where the parameter is
        defined.
        """
        x_prime_inputs = getattr(self, "_x_prime_input_parameters", [])
        if not isinstance(x_prime_inputs, (list, tuple, set)):
            x_prime_inputs = []
        if parameter in x_prime_inputs:
            if x_prime is None:
                raise RuntimeError(
                    f"Prime-space input `{parameter}` requested for "
                    f"{self.name} but no x_prime array was provided."
                )
            x_prime[parameter] = value
        else:
            x[parameter] = value
        return x, x_prime

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
