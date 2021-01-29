
import logging


from ..reparameterisations import (
    default_reparameterisations,
    RescaleToBounds,
    AnglePair
)


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
    def __init__(self, allowed_bounds=['upper'], allow_both=False, **kwargs):
        super().__init__(**kwargs)

        self.detect_edges_kwargs['allowed_bounds'] = allowed_bounds
        self.detect_edges_kwargs['allow_both'] = allow_both
        self.detect_edges_kwargs['x_range'] = \
            self.uniform_parameter_prior_bounds

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
                        x_prime, x_prime, log_j, p, pp, compute_radius,
                        detect_edge_prime=True, **kwargs)
            else:
                x_prime[pp], lj = \
                    self._rescale_to_bounds(x_prime[pp] - self.offsets[p], p)
                log_j += lj
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Map inputs to the physical space from the prime space"""
        for p, pp in zip(reversed(self.parameters),
                         reversed(self.prime_parameters)):
            x_prime[pp] = self.convert_from_uniform_parameter(x[p])
            if p in self.boundary_inversion:
                x, x_prime, log_j = self._reverse_inversion(
                    x, x_prime, log_j, p, pp, **kwargs)
            else:
                x[p], lj = self._inverse_rescale_to_bounds(x_prime[pp], p)
                x[p] += self.offsets[p]
                log_j += lj
        return x, x_prime, log_j


default_gw = {
    'distance': (DistanceReparameterisation, {}),
    'time': (RescaleToBounds, {'offset': True}),
    'sky-ra-dec': (AnglePair, {'convention': 'ra-dec'}),
    'sky-az-zen': (AnglePair, {'convention': 'az-zen'}),
    'mass_ratio': (RescaleToBounds, {'detect_edges': True,
                                     'boundary_inversion': True,
                                     'inversion_type': 'duplicate'}),
    'mass': (RescaleToBounds, {}),
}


default_gw.update(default_reparameterisations)
