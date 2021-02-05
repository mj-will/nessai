
import logging

from ..reparameterisations import (
    default_reparameterisations,
    Reparameterisation,
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
    if isinstance(reparameterisation, str):
        rc, kwargs = default_gw.get(reparameterisation, (None, None))
        if rc is None:
            raise ValueError(
                f'Unknown GW reparameterisation: {reparameterisation}')
        else:
            if kwargs is None:
                kwargs = {}
            else:
                kwargs = kwargs.copy()
            return rc, kwargs
    elif (isinstance(reparameterisation, type) and
            issubclass(reparameterisation, Reparameterisation)):
        return reparameterisation, {}
    else:
        raise RuntimeError('Reparmeterisation must a str or class that '
                           'inherits from `Reparameterisation`')


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
    conveter_kwargs : dict, optional
        Keyword arguments parsed to converter object that converts the distance
        to a parameter with a uniform prior.
    allowed_bounds : list, optional
        List of the allowed bounds for inversion
    kwargs :
        Additional kwargs are parsed to the parent class.
    """
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

        pre_rescaling = (self.distance_converter.to_uniform_parameter,
                         self.distance_converter.from_uniform_parameter)

        super().__init__(parameters=parameters, prior=prior,
                         prior_bounds=prior_bounds,
                         pre_rescaling=pre_rescaling, **kwargs)

        if self.distance_converter.has_conversion:
            self._prime_prior = log_uniform_prior
            self.has_prime_prior = True
            if not self.distance_converter.has_jacobian:
                logger.debug('Distance converter does not have Jacobian, '
                             'require prime prior')
                self.requires_prime_prior = True
            self.update_prime_prior_bounds()
        else:
            self.has_prime_prior = False

        self.detect_edges_kwargs['allowed_bounds'] = allowed_bounds
        self.detect_edges_kwargs['allow_both'] = allow_both
        self.detect_edges_kwargs['x_range'] = \
            self.prior_bounds[self.parameters[0]]


default_gw = {
    'distance': (DistanceReparameterisation, {'boundary_inversion': True,
                                              'detect_edges': True,
                                              'inversion_type': 'duplicate'}),
    'time': (RescaleToBounds, {'offset': True, 'update_bounds': True}),
    'sky-ra-dec': (AnglePair, {'convention': 'ra-dec'}),
    'sky-az-zen': (AnglePair, {'convention': 'az-zen'}),
    'mass_ratio': (RescaleToBounds, {'detect_edges': True,
                                     'boundary_inversion': True,
                                     'inversion_type': 'duplicate',
                                     'update_bounds': True}),
    'mass': (RescaleToBounds, {'update_bounds': True}),
}


default_gw.update(default_reparameterisations)
