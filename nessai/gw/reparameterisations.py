
import logging

import numpy as np

from ..reparameterisations import (
    default_reparameterisations,
    RescaleToBounds,
    AnglePair
)

from ..priors import log_uniform_prior

from .utils import get_distance_converter


logger = logging.getLogger(__name__)


def get_gw_reparameterisation(reparameterisation):
    """
    Get a reparameterisation from the default list plus specific GW
    classes.
    """
    rc, kwargs = default_gw.get(reparameterisation, (None, None))
    if rc is None:
        raise ValueError(
            f'Unknown GW reparameterisation: {reparameterisation}')
    else:
        return rc, kwargs


class DistanceReparameterisation(RescaleToBounds):
    def __init__(self, parameters=None, allowed_bounds=['upper'],
                 allow_both=False, converter_kwargs=None, prior=None,
                 prior_bounds=None, **kwargs):

        if isinstance(parameters, str):
            parameters = [parameters]

        if len(parameters) > 1:
            raise RuntimeError('Distance on supports one parameter')

        dc_class = get_distance_converter(prior)

        if converter_kwargs is None:
            converter_kwargs = {}
        self.distance_converter = dc_class(
            d_min=prior_bounds[parameters[0]][0],
            d_max=prior_bounds[parameters[0]][1],
            **converter_kwargs)

        super().__init__(parameters=parameters, prior=prior,
                         prior_bounds=prior_bounds, **kwargs)

        if self.distance_converter.has_conversion:
            self._prime_prior = log_uniform_prior
            self.has_prime_prior = True
            self.detect_edge_prime = True
            self.requires_prime_prior = True
        else:
            self.has_prime_prior = False
            self.detect_edge_prime = False

        self.orig_prior_bounds = self.prior_bounds.copy()
        self.prior_bounds = \
            {self.parameters[0]: self.convert_to_uniform_parameter(
                self.prior_bounds[self.parameters[0]])}

        self.detect_edges_kwargs['allowed_bounds'] = allowed_bounds
        self.detect_edges_kwargs['allow_both'] = allow_both
        self.detect_edges_kwargs['x_range'] = \
            self.prior_bounds[self.parameters[0]]

    def convert_to_uniform_parameter(self, d):
        """
        Convert to uniform distance parameter dc3
        """
        return self.distance_converter.to_uniform_parameter(d)

    def convert_from_uniform_parameter(self, d):
        """
        Convert from uniform distance parameter to luminosity_distance
        """
        return self.distance_converter.from_uniform_parameter(d)

    def reparameterise(self, x, x_prime, log_j, compute_radius=False,
                       **kwargs):
        """
        Rescale inputs to the prime space

        Parameters
        ----------
        x, x_prime :  array_like
            Arrays of samples in the physical and prime space
        log_j : array_like
            Array of values of log-Jacboian
        compute_radius : bool, optional
            If true force duplicate for inversion
        kwargs :
            Parsed to inversion function
        """
        for p, pp in zip(self.parameters, self.prime_parameters):
            x_prime[pp] = self.convert_to_uniform_parameter(x[p])
            if p in self.boundary_inversion:
                x, x_prime, log_j = self._apply_inversion(
                        x, x_prime, log_j, p, pp, compute_radius,
                        **kwargs)
            else:
                x_prime[pp], lj = \
                    self._rescale_to_bounds(x_prime[pp] - self.offsets[p], p)
                log_j += lj
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Map inputs to the physical space from the prime space"""
        for p, pp in zip(reversed(self.parameters),
                         reversed(self.prime_parameters)):
            if p in self.boundary_inversion:
                x, x_prime, log_j = self._reverse_inversion(
                    x, x_prime, log_j, p, pp, **kwargs)
            else:
                x[p], lj = self._inverse_rescale_to_bounds(x_prime[pp], p)
                x[p] += self.offsets[p]
                log_j += lj
            x[p] = self.convert_from_uniform_parameter(x[p])
        return x, x_prime, log_j

    def update_bounds(self, x):
        """Update the bounds used for the reparameterisation"""
        if self._update_bounds or self.bounds is None:
            up = self.convert_to_uniform_parameter(x[self.parameters[0]])
            self.bounds = \
                {p: [np.min(up - self.offsets[p]),
                     np.max(up - self.offsets[p])]
                 for p in self.parameters}
            logger.debug(f'New bounds: {self.bounds}')
            self.update_prime_prior_bounds()
        else:
            logger.debug('Update bounds not enabled')


default_gw = {
    'distance': (DistanceReparameterisation, {'boundary_inversion': True,
                                              'detect_edges': True,
                                              'inversion_type': 'duplicate'}),
    'time': (RescaleToBounds, {'offset': True, 'update_bounds': True}),
    'sky-ra-dec': (AnglePair, {'convention': 'ra-dec'}),
    'sky-az-zen': (AnglePair, {'convention': 'az-zen'}),
    'mass_ratio': (RescaleToBounds, {'detect_edges': True,
                                     'boundary_inversion': True,
                                     'inversion_type': 'duplicate'}),
    'mass': (RescaleToBounds, {'update_bounds': True}),
}


default_gw.update(default_reparameterisations)
