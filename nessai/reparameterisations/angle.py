# -*- coding: utf-8 -*-
"""
Reparameterisations for handling angles.
"""
import logging

import numpy as np
from scipy import stats

from .base import Reparameterisation
from ..priors import (
    log_2d_cartesian_prior,
    log_2d_cartesian_prior_sine,
)
from ..utils.rescaling import rescale_zero_to_one, inverse_rescale_zero_to_one

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
    prior : {'uniform', 'sine', None}
        Type of prior being used for sampling this angle. If specified, the
        prime prior is enabled. If None then it is disabled.
    """

    requires_bounded_prior = True

    def __init__(
        self, parameters=None, prior_bounds=None, scale=1.0, prior=None
    ):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        if len(self.parameters) == 1:
            self.parameters.append(self.parameters[0] + "_radial")
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

        self.prime_parameters = [self.angle + "_x", self.angle + "_y"]
        self.requires = []

        if prior in ["uniform", "sine"]:
            self.prior = prior
            self.has_prime_prior = True
            if self.prior == "uniform":
                self._prime_prior = log_2d_cartesian_prior
                self._k = self.scale * np.pi
            else:
                self._prime_prior = log_2d_cartesian_prior_sine
                self._k = np.pi
            logger.debug(f"Prime prior enabled for {self.name}")
        else:
            self.has_prime_prior = False
            logger.debug(f"Prime prior disabled for {self.name}")

    @property
    def angle(self):
        """The name of the angular parameter"""
        return self.parameters[0]

    @property
    def radial(self):
        """The name of the radial parameter"""
        return self.parameters[1]

    @property
    def radius(self):
        """The name of the radial parameter (equivalent to radial)"""
        return self.parameters[1]

    @property
    def x(self):
        """The name of x coordinate"""
        return self.prime_parameters[0]

    @property
    def y(self):
        """The name of y coordinate"""
        return self.prime_parameters[1]

    def _rescale_radial(self, x, x_prime, log_j, **kwargs):
        return x[self.parameters[1]], x, x_prime, log_j

    def _rescale_angle(self, x, x_prime, log_j, **kwargs):
        return x[self.parameters[0]] * self.scale, x, x_prime, log_j

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        return x, x_prime, log_j

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert the angle to Cartesian coordinates"""
        angle, x, x_prime, log_j = self._rescale_angle(
            x, x_prime, log_j, **kwargs
        )

        if self.chi:
            r = self.chi.rvs(size=x.size)
        else:
            r, x, x_prime, log_j = self._rescale_radial(
                x, x_prime, log_j, **kwargs
            )
        if any(r < 0):
            raise RuntimeError("Radius cannot be negative.")

        x_prime[self.prime_parameters[0]] = r * np.cos(angle)
        x_prime[self.prime_parameters[1]] = r * np.sin(angle)
        log_j += np.log(r)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to an angle and radial component"""
        x[self.parameters[1]] = np.sqrt(
            x_prime[self.prime_parameters[0]] ** 2
            + x_prime[self.prime_parameters[1]] ** 2
        )
        if self._zero_bound:
            x[self.parameters[0]] = (
                np.arctan2(
                    x_prime[self.prime_parameters[1]],
                    x_prime[self.prime_parameters[0]],
                )
                % (2.0 * np.pi)
                / self.scale
            )
        else:
            x[self.parameters[0]] = (
                np.arctan2(
                    x_prime[self.prime_parameters[1]],
                    x_prime[self.prime_parameters[0]],
                )
                / self.scale
            )

        log_j -= np.log(x[self.parameters[1]])
        x, x_prime, log_j = self._inverse_rescale_angle(x, x_prime, log_j)

        return x, x_prime, log_j

    def log_prior(self, x):
        """Prior for radial parameter"""
        return self.chi.logpdf(x[self.parameters[1]])

    def x_prime_log_prior(self, x_prime):
        """Compute the prior in the prime space assuming a uniform prior"""
        if self.has_prime_prior:
            return self._prime_prior(
                x_prime[self.prime_parameters[0]],
                x_prime[self.prime_parameters[1]],
                k=self._k,
            )
        else:
            raise RuntimeError("Prime prior")


class ToCartesian(Angle):
    """Convert a parameter to Cartesian coordinates"""

    def __init__(self, mode="split", scale=np.pi, **kwargs):
        super().__init__(scale=scale, **kwargs)

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
            x[self.parameters[0]], *self.prior_bounds[self.parameters[0]]
        )
        log_j += lj
        if self.mode == "duplicate" or compute_radius:
            angle = np.concatenate([angle, -angle])
            x = np.concatenate([x, x])
            x_prime = np.concatenate([x_prime, x_prime])
            log_j = np.concatenate([log_j, log_j])
        elif self.mode == "split":
            neg = np.random.choice(angle.size, angle.size // 2, replace=False)
            angle[neg] *= -1

        angle *= self.scale
        return angle, x, x_prime, log_j

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        x[self.parameters[0]], lj = inverse_rescale_zero_to_one(
            np.abs(x[self.parameters[0]]),
            *self.prior_bounds[self.parameters[0]],
        )
        log_j += lj
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
    prior : str, {'isotropic', None}
        Type of prior, used to enable use of the prime prior.
    convention : str, {'ra-dec', 'az-zen'}
        Convention used for defining the spherical polar coordinates. If not
        set, it will be guessed based on either dec or zen. Where it is assumed
        declination is defined on [-pi/2, pi/2] and zenith on [0, pi].
    """

    requires_bounded_prior = True
    known_priors = ["isotropic", None]
    _conventions = {"az-zen": [0, np.pi], "ra-dec": [-np.pi / 2, np.pi / 2]}

    def __init__(
        self, parameters=None, prior_bounds=None, prior=None, convention=None
    ):

        if len(parameters) not in [2, 3]:
            raise RuntimeError(
                "Must use a pair of angles or a pair plus a radius"
            )

        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        # Determine which parameter is for the horizontal plane
        # and which is for the vertical plane
        logger.debug("Checking order of parameters")
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
        else:
            parameters[0], parameters[1] = parameters[hz], parameters[vt]
            m = "_".join(parameters)
            parameters.append(f"{m}_radial")
            self.chi = stats.chi(3)
            self.has_prior = True

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

        self.parameters = parameters
        self.prime_parameters = [f"{m}_{x}" for x in ["x", "y", "z"]]

        if prior == "isotropic" and self.chi:
            self.has_prime_prior = True
            logger.info(f"Prime prior enabled for {self.name}")
        elif prior not in self.known_priors:
            raise ValueError(
                f"Unknown prior: `{prior}`. Choose from: {self.known_priors}"
            )
        else:
            self.has_prime_prior = False
            logger.debug(f"Prime prior disabled for {self.name}")

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
                    f"Could not determine convention for: {self.parameters}!"
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

        self.requires = []

    @property
    def angles(self):
        """Names of the two angles.

        Order is: angle along the horizon, vertical angle.
        """
        return self.parameters[:2]

    @property
    def radial(self):
        """Name of the radial parameter"""
        return self.parameters[-1]

    @property
    def x(self):
        """Name of the first Cartesian coordinate"""
        return self.prime_parameters[0]

    @property
    def y(self):
        """Name of the second Cartesian coordinate"""
        return self.prime_parameters[1]

    @property
    def z(self):
        """Name of the third Cartesian coordinate"""
        return self.prime_parameters[2]

    def _az_zen(self, x, x_prime, log_j, r):
        x_prime[self.prime_parameters[0]] = (
            r * np.sin(x[self.parameters[1]]) * np.cos(x[self.parameters[0]])
        )
        x_prime[self.prime_parameters[1]] = (
            r * np.sin(x[self.parameters[1]]) * np.sin(x[self.parameters[0]])
        )
        x_prime[self.prime_parameters[2]] = r * np.cos(x[self.parameters[1]])
        log_j += 2 * np.log(r) + np.log(np.sin(x[self.parameters[1]]))
        return x, x_prime, log_j

    def _ra_dec(self, x, x_prime, log_j, r):
        x_prime[self.prime_parameters[0]] = (
            r * np.cos(x[self.parameters[1]]) * np.cos(x[self.parameters[0]])
        )
        x_prime[self.prime_parameters[1]] = (
            r * np.cos(x[self.parameters[1]]) * np.sin(x[self.parameters[0]])
        )
        x_prime[self.prime_parameters[2]] = r * np.sin(x[self.parameters[1]])
        log_j += 2 * np.log(r) + np.log(np.cos(x[self.parameters[1]]))
        return x, x_prime, log_j

    def _inv_az_zen(self, x, x_prime, log_j):
        x[self.parameters[2]] = np.sqrt(
            x_prime[self.prime_parameters[0]] ** 2.0
            + x_prime[self.prime_parameters[1]] ** 2.0
            + x_prime[self.prime_parameters[2]] ** 2.0
        )

        if self._modulo_2pi:
            x[self.parameters[0]] = np.arctan2(
                x_prime[self.prime_parameters[1]],
                x_prime[self.prime_parameters[0]],
            ) % (2.0 * np.pi)
        else:
            x[self.parameters[0]] = np.arctan2(
                x_prime[self.prime_parameters[1]],
                x_prime[self.prime_parameters[0]],
            )
        x[self.parameters[1]] = np.arctan2(
            np.sqrt(
                x_prime[self.prime_parameters[0]] ** 2.0
                + x_prime[self.prime_parameters[1]] ** 2.0
            ),
            x_prime[self.prime_parameters[2]],
        )
        log_j += -2 * np.log(x[self.parameters[2]]) - np.log(
            np.sin(x[self.parameters[1]])
        )

        return x, x_prime, log_j

    def _inv_ra_dec(self, x, x_prime, log_j):
        x[self.parameters[2]] = np.sqrt(
            x_prime[self.prime_parameters[0]] ** 2.0
            + x_prime[self.prime_parameters[1]] ** 2.0
            + x_prime[self.prime_parameters[2]] ** 2.0
        )

        if self._modulo_2pi:
            x[self.parameters[0]] = np.arctan2(
                x_prime[self.prime_parameters[1]],
                x_prime[self.prime_parameters[0]],
            ) % (2.0 * np.pi)
        else:
            x[self.parameters[0]] = np.arctan2(
                x_prime[self.prime_parameters[1]],
                x_prime[self.prime_parameters[0]],
            )

        x[self.parameters[1]] = np.arctan2(
            x_prime[self.prime_parameters[2]],
            np.sqrt(
                x_prime[self.prime_parameters[0]] ** 2.0
                + x_prime[self.prime_parameters[1]] ** 2.0
            ),
        )

        log_j += -2 * np.log(x[self.parameters[2]]) - np.log(
            np.cos(x[self.parameters[1]])
        )
        return x, x_prime, log_j

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Convert the spherical polar angles to Cartesian coordinates
        """
        if self.chi:
            r = self.chi.rvs(size=x.size)
        else:
            r = x[self.radial]
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
            return self.chi.logpdf(x[self.parameters[2]])
        else:
            raise RuntimeError(
                "log_prior is not defined when a radial parameter has been "
                "specified!"
            )

    def x_prime_log_prior(self, x_prime):
        """
        Log probability of 3d Cartesian coordinates for an isotropic
        distribution of angles and a radial component drawn from a chi
        distribution with three degrees of freedom.
        """
        if self.has_prime_prior:
            return (
                -1.5 * np.log(2 * np.pi)
                - np.sum(
                    [x_prime[pp] ** 2 for pp in self.prime_parameters], axis=0
                )
                / 2
            )
        else:
            raise RuntimeError("x prime prior is not defined!")
