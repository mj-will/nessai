# -*- coding: utf-8 -*-
"""
Reparameterisations for handling angles.
"""

import logging

import numpy as np
from scipy import stats

from ..utils.rescaling import inverse_rescale_zero_to_one, rescale_zero_to_one
from .base import Reparameterisation

logger = logging.getLogger(__name__)


class Angle(Reparameterisation):
    """Reparameterisation for a single angle.

    This reparameterisations converts an angle to Cartesian coordinates
    using either a corresponding radial parameter or an auxiliary radial
    parameter. When using the auxiliary parameter, samples are drawn from
    a chi-distribution with two degrees of freedom.

    Parameters
    ----------
    parameters : str or list
        Parameter(s) to use for the reparameterisation. Either just an angle
        or an angle and corresponding radial parameter
    prior_bounds : dict
        Dictionary of prior bounds. Does not need to be specified when defining
        `reparameterisations`.
    scale : float, optional
        Value used to rescale the angle before converting to Cartesian
        coordinates. If None the scale will be set to 2pi / prior_bounds.
    """

    requires_bounded_prior = True

    def __init__(
        self,
        input_parameters=None,
        output_parameters=None,
        persistent_parameters=None,
        auxiliary_parameters=None,
        prior_bounds=None,
        scale=1.0,
        rng=None,
        inverse_input_parameters=None,
        parameters=None,
    ):
        super().__init__(
            input_parameters=input_parameters,
            output_parameters=output_parameters,
            persistent_parameters=persistent_parameters,
            auxiliary_parameters=auxiliary_parameters,
            prior_bounds=prior_bounds,
            rng=rng,
            inverse_input_parameters=inverse_input_parameters,
            parameters=parameters,
        )

        if len(self.parameters) == 1:
            self.auxiliary_parameters = [self.parameters[0] + "_radial"]
            self.chi = stats.chi(2)
            self.has_prior = True
        elif len(self.parameters) == 2:
            self.chi = False
        else:
            raise RuntimeError("Too many parameters for Angle")

        if scale is None:
            logger.debug("Scale is None, using 2pi / prior_range")
            self.scale = 2.0 * np.pi / np.ptp(self.prior_bounds[self.angle])
        else:
            self.scale = scale

        if prior_bounds[self.angle][0] == 0:
            self._zero_bound = True
        else:
            self._zero_bound = False

        # overwrite the prime parameters if not specified
        if output_parameters is None:
            self.output_parameters = [self.angle + "_x", self.angle + "_y"]

    @property
    def angle(self):
        """The name of the angular parameter"""
        return self.parameters[0]

    @property
    def radial(self):
        """The name of the radial parameter"""
        if self.chi:
            return self.auxiliary_parameters[0]
        return self.parameters[1]

    @property
    def radius(self):
        """The name of the radial parameter (equivalent to radial)"""
        return self.radial

    @property
    def x(self):
        """The name of x coordinate"""
        return self.output_parameters[0]

    @property
    def y(self):
        """The name of y coordinate"""
        return self.output_parameters[1]

    def _rescale_radial(self, x, x_prime, log_j, **kwargs):
        return (
            self.get_value(self.radial, x, x_prime),
            x,
            x_prime,
            log_j,
        )

    def _rescale_angle(self, x, x_prime, log_j, **kwargs):
        return (
            self.get_value(self.angle, x, x_prime) * self.scale,
            x,
            x_prime,
            log_j,
        )

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        return x, x_prime, log_j

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert the angle to Cartesian coordinates"""
        angle, x, x_prime, log_j = self._rescale_angle(
            x, x_prime, log_j, **kwargs
        )

        if self.chi:
            r = self.chi.rvs(size=x.size, random_state=self.rng)
        else:
            r, x, x_prime, log_j = self._rescale_radial(
                x, x_prime, log_j, **kwargs
            )
        if any(r < 0):
            raise RuntimeError("Radius cannot be negative.")

        x_prime[self.output_parameters[0]] = r * np.cos(angle)
        x_prime[self.output_parameters[1]] = r * np.sin(angle)
        log_j += np.log(r)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to an angle and radial component"""
        radial = np.sqrt(
            x_prime[self.output_parameters[0]] ** 2
            + x_prime[self.output_parameters[1]] ** 2
        )
        if self._zero_bound:
            angle = (
                np.arctan2(
                    x_prime[self.output_parameters[1]],
                    x_prime[self.output_parameters[0]],
                )
                % (2.0 * np.pi)
                / self.scale
            )
        else:
            angle = (
                np.arctan2(
                    x_prime[self.output_parameters[1]],
                    x_prime[self.output_parameters[0]],
                )
                / self.scale
            )

        log_j -= np.log(radial)
        x, x_prime = self._set_value(self.radial, radial, x, x_prime)
        x, x_prime = self._set_value(self.angle, angle, x, x_prime)
        x, x_prime, log_j = self._inverse_rescale_angle(x, x_prime, log_j)

        return x, x_prime, log_j

    def log_prior(self, x):
        """Prior for radial parameter"""
        return self.chi.logpdf(x[self.radial])


class ToCartesian(Angle):
    """Convert a parameter to Cartesian coordinates"""

    def __init__(self, mode="split", scale=np.pi, rng=None, **kwargs):
        super().__init__(scale=scale, rng=rng, **kwargs)

        self.mode = mode
        if self.mode not in ["duplicate", "split", "half"]:
            raise RuntimeError(f"Unknown mode: {self.mode}")
        logger.debug(f"Using mode: {self.mode}")

        self._zero_bound = False
        self._k = self.prior_bounds[self.parameters[0]][1]

    def _rescale_angle(
        self, x, x_prime, log_j, compute_radius=False, **kwargs
    ):
        angle, lj = rescale_zero_to_one(
            self.get_value(self.parameters[0], x, x_prime),
            *self.prior_bounds[self.parameters[0]],
        )
        log_j += lj
        if self.mode == "duplicate" or compute_radius:
            angle = np.concatenate([angle, -angle])
            x = np.concatenate([x, x])
            x_prime = np.concatenate([x_prime, x_prime])
            log_j = np.concatenate([log_j, log_j])
        elif self.mode == "split":
            neg = self.rng.choice(angle.size, angle.size // 2, replace=False)
            angle[neg] *= -1

        angle *= self.scale
        return angle, x, x_prime, log_j

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        angle, lj = inverse_rescale_zero_to_one(
            np.abs(self.get_value(self.parameters[0], x, x_prime)),
            *self.prior_bounds[self.parameters[0]],
        )
        log_j += lj
        x, x_prime = self._set_value(self.parameters[0], angle, x, x_prime)
        return x, x_prime, log_j


class AnglePair(Reparameterisation):
    """Reparameterisation for a pair of angles and a radial component.

    Converts to three-dimensional Cartesian coordinates.

    If the radial component is not specified, it is sampled from a chi-
    distribution with three degrees of freedom.

    Notes
    -----
    The parameters will be reordered such that the first parameter is the angle
    along the horizon, the second parameter is the vertical angle and the last
    parameter is the radial parameter.

    Parameters
    ----------
    parameters : list
        List of parameters. Must contain at least the two angles and,
        optionally, also a radial component.
    prior_bounds : dict
        Dictionary of prior bounds for each parameter
    convention : str, {'ra-dec', 'az-zen'}
        Convention used for defining the spherical polar coordinates. If not
        set, it will be guessed based on either dec or zen. Where it is assumed
        declination is defined on [-pi/2, pi/2] and zenith on [0, pi].
    """

    requires_bounded_prior = True
    _conventions = {"az-zen": [0, np.pi], "ra-dec": [-np.pi / 2, np.pi / 2]}

    def __init__(
        self,
        input_parameters=None,
        output_parameters=None,
        persistent_parameters=None,
        auxiliary_parameters=None,
        prior_bounds=None,
        convention=None,
        rng=None,
        inverse_input_parameters=None,
        parameters=None,
    ):
        inputs = (
            input_parameters if input_parameters is not None else parameters
        )
        if len(inputs) not in [2, 3]:
            raise RuntimeError(
                "Must use a pair of angles or a pair plus a radius"
            )

        super().__init__(
            input_parameters=input_parameters,
            output_parameters=output_parameters,
            persistent_parameters=persistent_parameters,
            auxiliary_parameters=auxiliary_parameters,
            prior_bounds=prior_bounds,
            rng=rng,
            inverse_input_parameters=inverse_input_parameters,
            parameters=parameters,
        )

        # Determine which parameter is for the horizontal plane
        # and which is for the vertical plane
        logger.debug("Checking order of parameters")
        parameters = self.input_parameters.copy()
        b = np.ptp([prior_bounds[p] for p in parameters], axis=1)
        try:
            hz = np.where(b == (2 * np.pi))[0][0]
            vt = np.where(b == np.pi)[0][0]
        except IndexError:
            raise ValueError(
                f"Invalid prior ranges: {self.prior_bounds}. "
                "Parameters must be defined over a range of pi and 2 pi!"
            )
        # Make sure order is horizon, vertical, radial
        if len(parameters) == 3:
            r = list({0, 1, 2} - {hz, vt})[0]
            parameters[0], parameters[1], parameters[2] = (
                parameters[hz],
                parameters[vt],
                parameters[r],
            )
            m = "_".join(parameters)
            self.chi = False
            auxiliary_parameters = []
        else:
            parameters[0], parameters[1] = parameters[hz], parameters[vt]
            m = "_".join(parameters)
            self.chi = stats.chi(3)
            self.has_prior = True
            auxiliary_parameters = [f"{m}_radial"]

        logger.debug(f"Parameters are: {parameters}")

        # Modulo 2pi is used if the first parameter (ra/az) in defined on
        # [0, 2pi] since output would otherwise be [-pi, pi]
        if np.array_equal(self.prior_bounds[parameters[0]], [0, 2 * np.pi]):
            self._modulo_2pi = True
        elif np.array_equal(self.prior_bounds[parameters[0]], [-np.pi, np.pi]):
            self._modulo_2pi = False
        else:
            raise ValueError(
                f"Prior bounds for {parameters[0]} must be [0, 2pi] or "
                f"[-pi, pi]. Received: {self.prior_bounds[parameters[0]]}"
            )

        self.input_parameters = parameters
        self.auxiliary_parameters = auxiliary_parameters
        self.output_parameters = [f"{m}_{x}" for x in ["x", "y", "z"]]

        if convention is None:
            logger.debug("Trying to determine convention")
            if np.array_equal(
                self.prior_bounds[self.parameters[1]], [0, np.pi]
            ):
                self.convention = "az-zen"
            elif np.array_equal(
                self.prior_bounds[self.parameters[1]], [-np.pi / 2, np.pi / 2]
            ):
                self.convention = "ra-dec"
            else:
                raise RuntimeError(
                    f"Could not determine convention for: {self.input_parameters}!"
                )
        elif convention in self._conventions.keys():
            self.convention = convention
            if not np.array_equal(
                self.prior_bounds[parameters[1]], self._conventions[convention]
            ):
                raise ValueError(
                    f"Prior bounds for {parameters[1]} must be "
                    f"{self._conventions[convention]} for the "
                    f"{convention} convention. "
                    f"Received: {self.prior_bounds[parameters[1]]}."
                )
        else:
            raise ValueError(
                f"Unknown convention: `{convention}`. "
                f"Choose from: {list(self._conventions.keys())}."
            )

        logger.debug(f"Using convention: {self.convention}")

        self._resolved_forward_inputs = False

    @property
    def angles(self):
        """Names of the two angles.

        Order is: angle along the horizon, vertical angle.
        """
        return self.parameters[:2]

    @property
    def radial(self):
        """Name of the radial parameter"""
        if self.chi:
            return self.auxiliary_parameters[0]
        return self.parameters[-1]

    @property
    def x(self):
        """Name of the first Cartesian coordinate"""
        return self.output_parameters[0]

    @property
    def y(self):
        """Name of the second Cartesian coordinate"""
        return self.output_parameters[1]

    @property
    def z(self):
        """Name of the third Cartesian coordinate"""
        return self.output_parameters[2]

    def _az_zen(self, x, x_prime, log_j, r):
        horizontal = self.get_value(self.parameters[0], x, x_prime)
        vertical = self.get_value(self.parameters[1], x, x_prime)
        x_prime[self.output_parameters[0]] = (
            r * np.sin(vertical) * np.cos(horizontal)
        )
        x_prime[self.output_parameters[1]] = (
            r * np.sin(vertical) * np.sin(horizontal)
        )
        x_prime[self.output_parameters[2]] = r * np.cos(vertical)
        log_j += 2 * np.log(r) + np.log(np.sin(vertical))
        return x, x_prime, log_j

    def _ra_dec(self, x, x_prime, log_j, r):
        horizontal = self.get_value(self.parameters[0], x, x_prime)
        vertical = self.get_value(self.parameters[1], x, x_prime)
        x_prime[self.output_parameters[0]] = (
            r * np.cos(vertical) * np.cos(horizontal)
        )
        x_prime[self.output_parameters[1]] = (
            r * np.cos(vertical) * np.sin(horizontal)
        )
        x_prime[self.output_parameters[2]] = r * np.sin(vertical)
        log_j += 2 * np.log(r) + np.log(np.cos(vertical))
        return x, x_prime, log_j

    def _inv_az_zen(self, x, x_prime, log_j):
        radial = np.sqrt(
            x_prime[self.output_parameters[0]] ** 2.0
            + x_prime[self.output_parameters[1]] ** 2.0
            + x_prime[self.output_parameters[2]] ** 2.0
        )

        if self._modulo_2pi:
            horizontal = np.arctan2(
                x_prime[self.output_parameters[1]],
                x_prime[self.output_parameters[0]],
            ) % (2.0 * np.pi)
        else:
            horizontal = np.arctan2(
                x_prime[self.output_parameters[1]],
                x_prime[self.output_parameters[0]],
            )
        vertical = np.arctan2(
            np.sqrt(
                x_prime[self.output_parameters[0]] ** 2.0
                + x_prime[self.output_parameters[1]] ** 2.0
            ),
            x_prime[self.output_parameters[2]],
        )
        log_j += -2 * np.log(radial) - np.log(np.sin(vertical))
        x, x_prime = self._set_value(self.radial, radial, x, x_prime)
        x, x_prime = self._set_value(
            self.parameters[0], horizontal, x, x_prime
        )
        x, x_prime = self._set_value(self.parameters[1], vertical, x, x_prime)

        return x, x_prime, log_j

    def _inv_ra_dec(self, x, x_prime, log_j):
        radial = np.sqrt(
            x_prime[self.output_parameters[0]] ** 2.0
            + x_prime[self.output_parameters[1]] ** 2.0
            + x_prime[self.output_parameters[2]] ** 2.0
        )

        if self._modulo_2pi:
            horizontal = np.arctan2(
                x_prime[self.output_parameters[1]],
                x_prime[self.output_parameters[0]],
            ) % (2.0 * np.pi)
        else:
            horizontal = np.arctan2(
                x_prime[self.output_parameters[1]],
                x_prime[self.output_parameters[0]],
            )

        vertical = np.arctan2(
            x_prime[self.output_parameters[2]],
            np.sqrt(
                x_prime[self.output_parameters[0]] ** 2.0
                + x_prime[self.output_parameters[1]] ** 2.0
            ),
        )

        log_j += -2 * np.log(radial) - np.log(np.cos(vertical))
        x, x_prime = self._set_value(self.radial, radial, x, x_prime)
        x, x_prime = self._set_value(
            self.parameters[0], horizontal, x, x_prime
        )
        x, x_prime = self._set_value(self.parameters[1], vertical, x, x_prime)
        return x, x_prime, log_j

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Convert the spherical polar angles to Cartesian coordinates
        """
        if self.chi:
            r = self.chi.rvs(size=x.size, random_state=self.rng)
        else:
            r = self.get_value(self.radial, x, x_prime)
        if any(r < 0):
            raise RuntimeError("Radius cannot be negative.")

        if self.convention == "az-zen":
            return self._az_zen(x, x_prime, log_j, r)
        else:
            return self._ra_dec(x, x_prime, log_j, r)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to spherical polar angles"""
        if self.convention == "az-zen":
            return self._inv_az_zen(x, x_prime, log_j)
        else:
            return self._inv_ra_dec(x, x_prime, log_j)

    def log_prior(self, x):
        """Prior for radial parameter"""
        if self.chi and self.has_prior:
            return self.chi.logpdf(x[self.radial])
        else:
            raise RuntimeError(
                "log_prior is not defined when a radial parameter has been "
                "specified!"
            )
