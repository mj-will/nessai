# -*- coding: utf-8 -*-
"""
Specific reparameterisations for gravitational-wave inference.
"""

import logging

import numpy as np

from ..reparameterisations import (
    default_reparameterisations,
    Reparameterisation,
    RescaleToBounds,
    AnglePair,
    get_reparameterisation,
)

from ..priors import log_uniform_prior

from .utils import get_distance_converter


logger = logging.getLogger(__name__)


def get_gw_reparameterisation(reparameterisation):
    """
    Get a reparameterisation from the default list plus specific GW
    classes.

    Parameters
    ----------
    reparameterisation : str, \
            :obj:`nessai.reparameterisations.Reparameterisation`
        Name of the reparameterisations to return or a class that inherits from
        :obj:`~nessai.reparameterisations.Reparameterisation`

    Returns
    -------
    :obj:`nessai.reparameteristaions.Reparameterisation`
        Reparameterisation class.
    dict
        Keyword arguments for the specific reparameterisation.
    """
    return get_reparameterisation(reparameterisation, defaults=default_gw)


class DistanceReparameterisation(RescaleToBounds):
    """Reparameterisation for distance.

    If the prior is specified and is one of the known priors then a rescaling
    is applied such that the resulting parameter has a uniform prior. If the
    prior is not specified, then the distance is rescaled an inversion is
    allowed on only the upper bound.

    Known priors
    ------------
    * Power-law: requires specifying the power. See converter kwargs.
    * Uniform-comoving-volume: uses a lookup table to convert to co-moving
    distance.

    Parameters
    ----------
    parameters : str
        Name of distance parameter to rescale.
    prior : {'power-law', 'uniform-comoving-volume'}, optional
        Prior used for the distance parameter
    prior_bounds : tuple
        Tuple of lower and upper bounds on the prior
    converter_kwargs : dict, optional
        Keyword arguments parsed to converter object that converts the distance
        to a parameter with a uniform prior.
    allowed_bounds : list, optional
        List of the allowed bounds for inversion
    kwargs :
        Additional kwargs are parsed to the parent class.
    """

    requires_bounded_prior = True

    def __init__(
        self,
        parameters=None,
        allowed_bounds=["upper"],
        allow_both=False,
        converter_kwargs=None,
        prior=None,
        prior_bounds=None,
        **kwargs,
    ):

        if isinstance(parameters, str):
            parameters = [parameters]

        if len(parameters) > 1:
            raise RuntimeError(
                "DistanceReparameterisation only supports one parameter"
            )

        dc_class = get_distance_converter(prior)

        if converter_kwargs is None:
            converter_kwargs = {}
        self.distance_converter = dc_class(
            d_min=prior_bounds[parameters[0]][0],
            d_max=prior_bounds[parameters[0]][1],
            **converter_kwargs,
        )

        pre_rescaling = (
            self.distance_converter.to_uniform_parameter,
            self.distance_converter.from_uniform_parameter,
        )

        super().__init__(
            parameters=parameters,
            prior=prior,
            prior_bounds=prior_bounds,
            pre_rescaling=pre_rescaling,
            **kwargs,
        )

        if self.distance_converter.has_conversion:
            self._prime_prior = log_uniform_prior
            self.has_prime_prior = True
            if not self.distance_converter.has_jacobian:
                logger.debug(
                    "Distance converter does not have Jacobian, "
                    "require prime prior"
                )
                self.requires_prime_prior = True
            self.update_prime_prior_bounds()
        else:
            self.has_prime_prior = False

        self.detect_edges_kwargs["allowed_bounds"] = allowed_bounds
        self.detect_edges_kwargs["allow_both"] = allow_both
        self.detect_edges_kwargs["x_range"] = self.prior_bounds[
            self.parameters[0]
        ]


class DeltaPhaseReparameterisation(Reparameterisation):
    """Reparameterisation that converts phase to delta phase.

    The Jacobian determinant of this transformation is 1.

    Requires "psi" and "theta_jn".

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name(s) of the parameter(s).
    prior_bounds : Union[list, dict]
        Prior bounds for the parameters
    """

    def __init__(self, parameters=None, prior_bounds=None):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)
        self.requires = ["psi", "theta_jn"]
        self.prime_parameters = ["delta_phase"]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
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
        x_prime[self.prime_parameters[0]] = (
            x[self.parameters[0]] + np.sign(np.cos(x["theta_jn"])) * x["psi"]
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

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        x[self.parameters[0]] = np.mod(
            x_prime[self.prime_parameters[0]]
            - np.sign(np.cos(x["theta_jn"])) * x["psi"],
            2 * np.pi,
        )
        return x, x_prime, log_j


class LISAExtrinsicSymmetry(Reparameterisation):

    requires_bounded_prior = True
    one_to_one = False
    lambda_bins = (np.pi / 2) * np.arange(4)
    beta_bins = np.array([-np.pi / 2, 0.0, np.pi / 2])

    _lambda_parameter = None
    _beta_parameter = None
    _psi_parameter = None
    _iota_parameter = None

    known_lambda_parameters = frozenset(
        [
            "eclipticlongitude",
        ]
    )
    known_beta_parameters = frozenset(
        [
            "eclipticlatitude",
        ]
    )
    known_psi_parameters = frozenset(
        [
            "polarization",
        ]
    )
    known_iota_parameters = frozenset(
        [
            "iota",
        ]
    )

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        lambda_parameter=None,
        beta_parameter=None,
        psi_parameter=None,
        iota_parameter=None,
    ):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)
        self.lambda_parameter = lambda_parameter
        self.beta_parameter = beta_parameter
        self.psi_parameter = psi_parameter
        self.iota_parameter = iota_parameter

    def determine_parameter(self, known_parameters):
        params = set(self.parameters)
        names = known_parameters.intersection(params)
        if len(names) > 1:
            raise RuntimeError("Multiple parameters match")
        elif not names:
            raise RuntimeError("No parameters match")
        else:
            name = names[0]
        return name

    @property
    def lambda_parameter(self):
        return self._lambda_parameter

    @lambda_parameter.setter
    def lambda_parameter(self, name):
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_lambda_parameters)
        self._lambda_parameter = name

    @property
    def beta_parameter(self):
        return self._beta_parameter

    @beta_parameter.setter
    def beta_parameter(self, name):
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_beta_parameters)
        self._beta_parameter = name

    @property
    def psi_parameter(self):
        return self._psi_parameter

    @psi_parameter.setter
    def psi_parameter(self, name):
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_psi_parameters)
        self._psi_parameter = name

    @property
    def iota_parameter(self):
        return self._iota_parameter

    @iota_parameter.setter
    def iota_parameter(self, name):
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_iota_parameters)
        self._iota_parameter = name

    def reparameterise(self, x, x_prime, log_j):
        long_num = np.digitize(x[self.lambda_parameter], self.lambda_bins) - 1
        lat_num = np.digitize(x[self.beta_parameter], self.beta_bins) - 1
        x_prime[self.psi_parameter] = np.mod(
            x[self.psi_parameter] - (long_num * 0.5 * np.pi),
            np.pi,
        )
        x_prime[self.psi_parameter] = np.where(
            lat_num,
            x_prime[self.psi_parameter],
            np.pi - x_prime[self.psi_parameter],
        )
        x_prime[self.iota_parameter] = np.where(
            lat_num,
            x[self.iota_parameter],
            np.pi - x[self.iota_parameter],
        )
        x_prime[self.beta_parameter] = np.where(
            lat_num,
            x[self.beta_parameter],
            -x[self.beta_parameter],
        )
        x_prime[self.lambda_parameter] = np.mod(
            x[self.lambda_parameter] - long_num * 0.5 * np.pi, 2 * np.pi
        )
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j):

        sky_num = np.random.choice(8, size=x.size)
        long_num = sky_num % 4
        lat_num = sky_num // 4

        x[self.lambda_parameter] = np.mod(
            x_prime[self.lambda_parameter] + long_num * 0.5 * np.pi, 2 * np.pi
        )
        x[self.beta_parameter] = np.where(
            lat_num,
            -x_prime[self.beta_parameter],
            x_prime[self.beta_parameter],
        )
        x[self.iota_parameter] = np.where(
            lat_num,
            np.pi - x_prime[self.iota_parameter],
            x_prime[self.iota_parameter],
        )
        x[self.psi_parameter] = np.where(
            lat_num,
            np.pi - x_prime[self.psi_parameter],
            x_prime[self.psi_parameter],
        )
        x[self.psi_parameter] = np.mod(
            x[self.psi_parameter] + (long_num * 0.5 * np.pi),
            np.pi,
        )
        return x, x_prime, log_j


default_gw = {
    "distance": (
        DistanceReparameterisation,
        {
            "boundary_inversion": True,
            "detect_edges": True,
            "inversion_type": "duplicate",
        },
    ),
    "time": (RescaleToBounds, {"offset": True, "update_bounds": True}),
    "sky-ra-dec": (AnglePair, {"convention": "ra-dec"}),
    "sky-az-zen": (AnglePair, {"convention": "az-zen"}),
    "mass_ratio": (
        RescaleToBounds,
        {
            "detect_edges": True,
            "boundary_inversion": True,
            "inversion_type": "duplicate",
            "update_bounds": True,
        },
    ),
    "mass": (RescaleToBounds, {"update_bounds": True}),
    "delta_phase": (DeltaPhaseReparameterisation, {}),
    "delta-phase": (DeltaPhaseReparameterisation, {}),
    "lisa-sky": (LISAExtrinsicSymmetry, {}),
    "lisa_sky": (LISAExtrinsicSymmetry, {}),
    "lisa-extrinsic": (LISAExtrinsicSymmetry, {}),
    "lisa_extrinsic": (LISAExtrinsicSymmetry, {}),
}


default_gw.update(default_reparameterisations)
