# -*- coding: utf-8 -*-
"""
Reparameterisations that rescale the parameters.
"""
import logging

import numpy as np

from .base import Reparameterisation
from ..priors import log_uniform_prior
from ..utils.rescaling import (
    configure_edge_detection,
    determine_rescaled_bounds,
    detect_edge,
    rescaling_functions,
    rescale_minus_one_to_one,
    rescale_zero_to_one,
    inverse_rescale_minus_one_to_one,
    inverse_rescale_zero_to_one,
)

logger = logging.getLogger(__name__)


class ScaleAndShift(Reparameterisation):
    """Reparameterisation that shifts and scales by a value.

    Applies

    .. math::
        x' = (x - shift) / scale

    Can apply the Z-score rescaling if :code:`estimate_scale` and
    :code:`estimate_shift` are both enabled.

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name of parameters to reparameterise.
    prior_bounds : list, dict or None
        Prior bounds for the parameter(s).
    scale : Optional[float]
        Scaling constant. If not specified, :code:`estimate_scale` must be
        True.
    shift : Optional[float]
        Shift constant. If not specified, no shift is applied.
    estimate_scale : bool
        If true, the value of :code:`scale` will be ignored and the standard
        deviation of the data will be used.
    estimate_shift : bool
        If true, the value of :code:`shift` will be ignored and the standard
        deviation of the data will be used.
    """

    requires_bounded_prior = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        scale=None,
        shift=None,
        estimate_scale=False,
        estimate_shift=False,
    ):
        if scale is None and not estimate_scale:
            raise RuntimeError("Must specify a scale or enable estimate_scale")
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        self.estimate_scale = estimate_scale
        self.estimate_shift = estimate_shift

        if self.estimate_scale or self.estimate_shift:
            self._update = True
        else:
            self._update = False

        if self.estimate_scale:
            self.scale = {p: None for p in parameters}
        elif scale:
            self.scale = self._check_value(scale, "scale")

        if self.estimate_shift:
            self.shift = {p: None for p in parameters}
        elif shift:
            self.shift = self._check_value(shift, "shift")
        else:
            self.shift = None

    def _check_value(self, value, name):
        """Helper function to check the scale or shift parameters are valid."""
        if isinstance(value, (int, float)):
            value = {p: float(value) for p in self.parameters}
        elif isinstance(value, list):
            if not len(value) == len(self.parameters):
                raise RuntimeError(
                    f"{name} list is a different length to the parameters."
                )
            value = {p: float(s) for p, s in zip(self.parameters, value)}
        elif isinstance(value, dict):
            if not set(self.parameters) == set(value.keys()):
                raise RuntimeError(f"Mismatched parameters with {name} dict")
            value = {p: float(s) for p, s in value.items()}
        else:
            raise TypeError(
                f"{name} input must be an instance of int, list or dict"
            )
        return value

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
        for p, pp in zip(self.parameters, self.prime_parameters):
            if self.shift:
                x_prime[pp] = (x[p] - self.shift[p]) / self.scale[p]
            else:
                x_prime[pp] = x[p] / self.scale[p]
            log_j -= np.log(np.abs(self.scale[p]))
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
        for p, pp in zip(self.parameters, self.prime_parameters):
            if self.shift:
                x[p] = (x_prime[pp] * self.scale[p]) + self.shift[p]
            else:
                x[p] = x_prime[pp] * self.scale[p]
            log_j += np.log(np.abs(self.scale[p]))
        return x, x_prime, log_j

    def update(self, x):
        """Update the scale and shift parameters if enabled."""
        if self._update:
            logger.debug("Updating scale and shift")
            for p in self.parameters:
                if self.estimate_scale:
                    self.scale[p] = np.std(x[p])
                if self.estimate_shift:
                    self.shift[p] = np.mean(x[p])


class Rescale(ScaleAndShift):
    """Reparameterisation that rescales the parameters by a constant factor
    that does not depend on the prior bounds.
    """


class RescaleToBounds(Reparameterisation):
    """Reparameterisation that maps to the specified interval.

    By default the interval is [-1, 1]. Also includes options for
    boundary inversion.

    This reparameterisation can handle multiple parameters.

    Parameters
    ----------
    parameters : list of str
        List of the names of parameters
    prior_bounds : dict
        Dictionary of prior bounds for each parameter. Does not need to be
        specified by the user.
    rescale_bounds : list of tuples, optional
        Bounds to rescale to.
    update_bounds : bool, optional
        Enable or disable updating bounds.
    prior : {'uniform', None}
        Type of prior used, if uniform prime prior is enabled.
    boundary_inversion : bool, list, dict, optional
        Configuration for boundary inversion. If a list, inversion is only
        applied to the parameters in the list based on `inversion_type`. If
        a `dict` then each item should be a parameter and a corresponding
        inversion type `{'split', 'inversion'}`.
    detect_edges : bool, optional
        Enable or disable edge detection for inversion.
    detect_edges_kwargs : dict, optional
        Dictionary of kwargs used to configure edge detection.
    offset : bool, optional
        Enable or disable offset subtraction. If `True` then the mean value
        of the prior is subtract of the parameter before the rescaling is
        applied. This is computed and applied after the 'pre-rescaling' if it
        has been specified.
    pre_rescaling : tuple of functions
        A function that applies a rescaling prior to the main rescaling and
        its inverse. Each function should return a value and the log-Jacobian
        determinant.
    post_rescaling : tuple of functions or {'logit}
        A function that applies a rescaling after to the main rescaling and
        its inverse. Each function should return a value and the log-Jacobian
        determinant. For example applying a logit after rescaling to [0, 1].
    """

    requires_bounded_prior = True

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        prior=None,
        rescale_bounds=None,
        boundary_inversion=None,
        detect_edges=False,
        inversion_type="split",
        detect_edges_kwargs=None,
        offset=False,
        update_bounds=True,
        pre_rescaling=None,
        post_rescaling=None,
    ):

        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        self.bounds = None
        self._edges = None
        self.detect_edge_prime = False

        self.has_pre_rescaling = True
        self.has_post_rescaling = True

        if rescale_bounds is None:
            logger.debug("Using default rescale bounds: [-1, 1]")
            self.rescale_bounds = {p: [-1, 1] for p in self.parameters}
        else:
            if isinstance(rescale_bounds, list):
                self.rescale_bounds = {
                    p: rescale_bounds for p in self.parameters
                }
            elif isinstance(rescale_bounds, dict):
                s = set(parameters) - set(rescale_bounds.keys())
                if s:
                    raise RuntimeError(
                        f"Missing rescale bounds for parameters: {s}"
                    )
                self.rescale_bounds = rescale_bounds
            else:
                raise TypeError(
                    "rescale_bounds must be an instance of list or dict. "
                    f"Got type: {type(rescale_bounds).__name__}"
                )

        if boundary_inversion is not None:
            if isinstance(boundary_inversion, list):
                self.boundary_inversion = {
                    n: inversion_type for n in boundary_inversion
                }
            elif isinstance(boundary_inversion, dict):
                self.boundary_inversion = boundary_inversion
            elif isinstance(boundary_inversion, bool):
                if boundary_inversion:
                    self.boundary_inversion = {
                        p: inversion_type for p in self.parameters
                    }
                else:
                    self.boundary_inversion = False
            else:
                raise TypeError(
                    "boundary_inversion must be a list, dict or bool. "
                    f"Got type: {type(boundary_inversion).__name__}"
                )
        else:
            self.boundary_inversion = False

        if self.boundary_inversion:
            for p in self.boundary_inversion:
                self.rescale_bounds[p] = [0, 1]

        self._update_bounds = update_bounds if not detect_edges else True
        self.detect_edges = detect_edges
        if self.boundary_inversion:
            self._edges = {n: None for n in self.parameters}
            self.detect_edges_kwargs = configure_edge_detection(
                detect_edges_kwargs, self.detect_edges
            )

        if self.detect_edges and not self.boundary_inversion:
            raise RuntimeError(
                "Must enable boundary inversion to use detect edges"
            )

        if prior == "uniform":
            self.prior = "uniform"
            self.has_prime_prior = True
            self._prime_prior = log_uniform_prior
            logger.debug(f"Prime prior enabled for {self.name}")
        else:
            self.has_prime_prior = False
            logger.debug(f"Prime prior disabled for {self.name}")

        self.configure_pre_rescaling(pre_rescaling)
        self.configure_post_rescaling(post_rescaling)

        if offset:
            self.offsets = {
                p: self.pre_rescaling(b[0])[0]
                + np.ptp(self.pre_rescaling(b)[0]) / 2
                for p, b in self.prior_bounds.items()
            }
            logger.debug(f"Offsets: {self.offsets}")
        else:
            self.offsets = {p: 0.0 for p in self.prior_bounds.keys()}

        self.set_bounds(self.prior_bounds)

    def configure_pre_rescaling(self, pre_rescaling):
        """Configure the rescaling applied before the standard rescaling.

        Used in :code:`DistanceReparameterisation`.

        Parameters
        ----------
        pre_rescaling : str or Tuple[Callable, Callable]
            Name of the pre-rescaling of tuple contain the forward and inverse
            functions that should return the rescaled value and the Jacobian.
        """
        if pre_rescaling is not None:
            if isinstance(pre_rescaling, str):
                logger.debug(f"Getting pre-rescaling function {pre_rescaling}")
                (
                    self.pre_rescaling,
                    self.pre_rescaling_inv,
                ) = rescaling_functions.get(
                    pre_rescaling.lower(), (None, None)
                )
                if self.pre_rescaling is None:
                    raise RuntimeError(
                        f"Unknown rescaling function: {pre_rescaling}"
                    )
            elif len(pre_rescaling) == 2:
                self.pre_rescaling = pre_rescaling[0]
                self.pre_rescaling_inv = pre_rescaling[1]
            else:
                raise RuntimeError(
                    "Pre-rescaling must be a str or tuple of two functions"
                )
            self.has_pre_rescaling = True
        else:
            logger.debug("No pre-rescaling to configure")
            self.has_pre_rescaling = False

    def configure_post_rescaling(self, post_rescaling):
        """Configure the rescaling applied after the standard rescaling.

        Used to apply the logit/sigmoid transforms after rescaling to [0, 1]

        Parameters
        ----------
        post_rescaling : str or Tuple[Callable, Callable]
            Name of the post-rescaling of tuple contain the forward and inverse
            functions that should return the rescaled value and the Jacobian.
        """
        if post_rescaling is not None:
            if isinstance(post_rescaling, str):
                logger.debug(
                    f"Getting post-rescaling function {post_rescaling}"
                )
                (
                    self.post_rescaling,
                    self.post_rescaling_inv,
                ) = rescaling_functions.get(
                    post_rescaling.lower(), (None, None)
                )
                if self.post_rescaling is None:
                    raise RuntimeError(
                        f"Unknown rescaling function: {post_rescaling}"
                    )
            elif len(post_rescaling) == 2:
                self.post_rescaling = post_rescaling[0]
                self.post_rescaling_inv = post_rescaling[1]
            else:
                raise RuntimeError(
                    "Post-rescaling must be a str or tuple of two functions"
                )
            logger.debug("Disabling prime prior with post-rescaling")
            self.has_prime_prior = False

            if post_rescaling == "logit":
                if self._update_bounds:
                    raise RuntimeError("Cannot use logit with update bounds")
                logger.debug("Setting bounds to [0, 1] for logit")
                self.rescale_bounds = {p: [0, 1] for p in self.parameters}
            self.has_post_rescaling = True
        else:
            logger.debug("No post-rescaling to configure")
            self.has_post_rescaling = False

    def pre_rescaling(self, x):
        """Function applied before rescaling to bounds"""
        return x.copy(), np.zeros_like(x)

    def pre_rescaling_inv(self, x):
        """Inverse of function applied before rescaling to bounds"""
        return x.copy(), np.zeros_like(x)

    def post_rescaling(self, x):
        """Function applied after rescaling to bounds"""
        return x, np.zeros_like(x)

    def post_rescaling_inv(self, x):
        """Inverse of function applied after rescaling to bounds"""
        return x, np.zeros_like(x)

    def _rescale_to_bounds(self, x, n):
        out = (
            self._rescale_factor[n]
            * (
                (x - self.bounds[n][0])
                / (self.bounds[n][1] - self.bounds[n][0])
            )
            + self._rescale_shift[n]
        )

        log_j = -np.log(self.bounds[n][1] - self.bounds[n][0]) + np.log(
            self._rescale_factor[n]
        )
        return out, log_j

    def _inverse_rescale_to_bounds(self, x, n):
        out = (self.bounds[n][1] - self.bounds[n][0]) * (
            x - self._rescale_shift[n]
        ) / self._rescale_factor[n] + self.bounds[n][0]

        log_j = np.log(self.bounds[n][1] - self.bounds[n][0]) - np.log(
            self._rescale_factor[n]
        )

        return out, log_j

    def _apply_inversion(
        self, x, x_prime, log_j, p, pp, compute_radius, test=None
    ):
        if self._edges[p] is None:
            self._edges[p] = detect_edge(
                x_prime[pp], test=test, **self.detect_edges_kwargs
            )
            self.update_prime_prior_bounds()

        if self._edges[p]:
            logger.debug(f"Apply inversion for {p} to {self._edges[p]} bound")
            logger.debug("Fixing bounds to [0, 1]")
            logger.debug("Rescaling")
            x_prime[pp], lj = rescale_zero_to_one(
                x_prime[pp] - self.offsets[p], *self.bounds[p]
            )
            log_j += lj
            if self._edges[p] == "upper":
                x_prime[pp] = 1 - x_prime[pp]
            if self.boundary_inversion[p] == "duplicate" or compute_radius:
                logger.debug("Inverting with duplication")
                x_inv = x_prime.copy()
                x_inv[pp] *= -1
                x_prime = np.concatenate([x_prime, x_inv])
                x = np.concatenate([x, x])
                log_j = np.concatenate([log_j, log_j])
            else:
                logger.debug("Inverting with splitting")
                inv = np.random.choice(
                    x_prime.size, x_prime.size // 2, replace=False
                )
                x_prime[pp][inv] *= -1
        else:
            logger.debug(f"Not using inversion for {p}")
            logger.debug(f"Rescaling to {self.rescale_bounds[p]}")
            x_prime[pp], lj = rescale_minus_one_to_one(
                x_prime[pp] - self.offsets[p],
                xmin=self.bounds[p][0],
                xmax=self.bounds[p][1],
            )

            log_j += lj

        return x, x_prime, log_j

    def _reverse_inversion(self, x, x_prime, log_j, p, pp):
        if self._edges[p]:
            inv = x[p] < 0.0
            x[p][~inv] = x[p][~inv]
            x[p][inv] = -x[p][inv]

            if self._edges[p] == "upper":
                x[p] = 1 - x[p]
            x[p], lj = inverse_rescale_zero_to_one(x[p], *self.bounds[p])
            x[p] += self.offsets[p]
            log_j += lj
        else:
            x[p], lj = inverse_rescale_minus_one_to_one(
                x[p], xmin=self.bounds[p][0], xmax=self.bounds[p][1]
            )
            x[p] += self.offsets[p]
            log_j += lj
        return x, x_prime, log_j

    def reparameterise(
        self, x, x_prime, log_j, compute_radius=False, **kwargs
    ):
        """
        Rescale inputs to the prime space

        Parameters
        ----------
        x, x_prime :  array_like
            Arrays of samples in the physical and prime space
        log_j : array_like
            Array of values of log-Jacobian
        compute_radius : bool, optional
            If true force duplicate for inversion
        kwargs :
            Parsed to inversion function
        """
        for p, pp in zip(self.parameters, self.prime_parameters):
            if self.has_pre_rescaling:
                x_prime[pp], lj = self.pre_rescaling(x[p])
                log_j += lj
            else:
                x_prime[pp] = x[p]

            if self.boundary_inversion and p in self.boundary_inversion:
                x, x_prime, log_j = self._apply_inversion(
                    x, x_prime, log_j, p, pp, compute_radius, **kwargs
                )
            else:
                x_prime[pp], lj = self._rescale_to_bounds(
                    x_prime[pp] - self.offsets[p], p
                )
                log_j += lj
            if self.has_post_rescaling:
                x_prime[pp], lj = self.post_rescaling(x_prime[pp])
                log_j += lj
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Map inputs to the physical space from the prime space"""
        for p, pp in zip(
            reversed(self.parameters), reversed(self.prime_parameters)
        ):
            if self.has_post_rescaling:
                x[p], lj = self.post_rescaling_inv(x_prime[pp])
                log_j += lj
            else:
                x[p] = x_prime[pp]
            if self.boundary_inversion and p in self.boundary_inversion:
                x, x_prime, log_j = self._reverse_inversion(
                    x, x_prime, log_j, p, pp, **kwargs
                )
            else:
                x[p], lj = self._inverse_rescale_to_bounds(x[p], p)
                x[p] += self.offsets[p]
                log_j += lj
            if self.has_pre_rescaling:
                x[p], lj = self.pre_rescaling_inv(x[p])
                log_j += lj
        return x, x_prime, log_j

    def reset_inversion(self):
        """Reset the edges for inversion"""
        if self._edges:
            self._edges = {n: None for n in self.parameters}

    def set_bounds(self, prior_bounds):
        """Set the initial bounds for rescaling"""
        self._rescale_factor = {
            p: np.ptp(self.rescale_bounds[p]) for p in self.parameters
        }
        self._rescale_shift = {
            p: self.rescale_bounds[p][0] for p in self.parameters
        }

        self.pre_prior_bounds = {
            p: self.pre_rescaling(prior_bounds[p])[0] for p in self.parameters
        }
        self.bounds = {
            p: self.pre_rescaling(b)[0] - self.offsets[p]
            for p, b in prior_bounds.items()
        }
        logger.debug(f"Initial bounds: {self.bounds}")
        self.update_prime_prior_bounds()

    def update_bounds(self, x):
        """Update the bounds used for the reparameterisation"""
        if self._update_bounds:
            self.bounds = {
                p: [
                    self.pre_rescaling(np.min(x[p]))[0] - self.offsets[p],
                    self.pre_rescaling(np.max(x[p]))[0] - self.offsets[p],
                ]
                for p in self.parameters
            }
            logger.debug(f"New bounds: {self.bounds}")
            self.update_prime_prior_bounds()

        else:
            logger.debug("Update bounds not enabled")

    def update_prime_prior_bounds(self):
        """Update the prior bounds used for the prime prior"""
        if self.has_prime_prior:
            self.prime_prior_bounds = {
                pp: self.post_rescaling(
                    np.asarray(
                        determine_rescaled_bounds(
                            self.pre_prior_bounds[p][0],
                            self.pre_prior_bounds[p][1],
                            self.bounds[p][0],
                            self.bounds[p][1],
                            invert=self._edges[p] if self._edges else None,
                            inversion=(
                                p in self.boundary_inversion
                                if self.boundary_inversion
                                else False
                            ),
                            offset=self.offsets[p],
                            rescale_bounds=self.rescale_bounds[p],
                        )
                    )
                )[0]
                for p, pp in zip(self.parameters, self.prime_parameters)
            }
            logger.debug(f"New prime bounds: {self.prime_prior_bounds}")

    def update(self, x):
        """Update the reparameterisation given some points.

        Includes resetting the inversions and updating the bounds.
        """
        self.update_bounds(x)
        self.reset_inversion()

    def x_prime_log_prior(self, x_prime):
        """Compute the prior in the prime space assuming a uniform prior"""
        if self.has_prime_prior:
            log_p = 0
            for pp in self.prime_parameters:
                log_p += self._prime_prior(
                    x_prime[pp], *self.prime_prior_bounds[pp]
                )
            return log_p
        else:
            raise RuntimeError(
                f"Prime prior is not configured for {self.name}"
            )
