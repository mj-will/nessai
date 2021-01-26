import logging

import numpy as np
from scipy import stats

from .utils import (
    detect_edge,
    rescale_zero_to_one,
    inverse_rescale_zero_to_one
)


logger = logging.getLogger(__name__)


def get_reparameterisation(reparameterisation):
    """Function to get a reparmeterisation class from a name"""
    rc = REPARAMETERISATION_DICT.get(reparameterisation, None)
    if rc is None:
        raise RuntimeError(f'Unknown reparameterisation: {reparameterisation}')
    else:
        return rc


class Reparameterisation:
    """
    Base object for reparameterisations.

    Parameters
    ----------
    parameters : str or list
        Name of parameters to reparameterise.
    """
    def __init__(self, parameters=None, prior_bounds=None):
        if not isinstance(parameters, (str, list)):
            raise TypeError('Parameters must be a str or list.')

        self.parameters = \
            [parameters] if isinstance(parameters, str) else parameters

        if set(parameters) - set(prior_bounds.keys()):
            raise RuntimeError('Mismatch between parameters and prior bounds')
        self.prior_bounds = prior_bounds
        self.prime_parameters = [p + '_prime' for p in self.parameters]
        self.requires = []
        logger.debug(f'Initialised reparameterisation: {self.name}')

    @property
    def name(self):
        """Unique name of the reparameterisations"""
        return self.__class__.__name__.lower() + '_' + \
            '_'.join(self.parameters)

    def reparameterise(self, x, x_prime, log_j):
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
        log_j : Log jacobian to be updated
        """
        raise NotImplementedError


class CombinedReparameterisation(dict):
    """Class to handle mulitple reparameterisations

    """
    def __init__(self, reparameterisations=[]):
        super().__init__()
        self.reparmeterisations = {}
        self.parameters = []
        self.prime_parameters = []
        self.requires = []
        self.add_reparameterisations(reparameterisations)

    def _add_reparameterisation(self, reparameterisation):
        if ((r := reparameterisation.requires) and
                (r not in self.parameters or r not in self.prime_parameters)):
            raise RuntimeError(
                f'Could not add {reparameterisation}, missing requirement(s): '
                f'{reparameterisation.requires}.')

        self[reparameterisation.name] = reparameterisation
        self.parameters += reparameterisation.parameters
        self.prime_parameters += reparameterisation.prime_parameters
        self.requires += reparameterisation.requires

    def add_reparameterisations(self, reparameterisations):
        """Add multiple reparameterisations

        Parameters
        ----------
        reparameterisations : list of :`obj`:Reparameterisation
            List of reparameterisations to add.
        """
        if not isinstance(reparameterisations, list):
            reparameterisations = [reparameterisations]
        for r in reparameterisations:
            self._add_reparameterisation(r)

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
        for r in self.values():
            x, x_prime, log_j = r.reparameterise(x, x_prime, log_j, **kwargs)
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
        for r in reversed(self.values()):
            x, x_prime, log_j = r.inverse_reparameterise(
                x, x_prime, log_j, **kwargs)
        return x, x_prime, log_j

    def update_bounds(self, x):
        """
        Update the bounds used for the reparameterisation
        """
        for r in self.values():
            try:
                logger.debug(f'Updating bounds for: {r.name}')
                r.update_bounds(x)
            except Exception as e:
                print(e)

    def reset_inversion(self):
        """
        Reset edges for boundary inversion
        """
        for r in self.values():
            try:
                r.reset_inversion()
            except Exception as e:
                print(e)


class NullReparameterisation(Reparameterisation):

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
        x_prime[self.prime_parameters] = x[self.parameters]
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
        x[self.parameters] = x[self.prime_parameters]
        return x, x_prime, log_j


class RescaleToBounds(Reparameterisation):
    """Reparmeterisation that maps to the specified interval.

    By default the interval in [-1, 1]

    Parameters
    ----------
    parameters : list of str
        List of the names of parameters
    prior_bounds : dict
        Dictionary of prior bounds for each parameter
    rescale_bounds : list of tuples
        Bounds to rescale to
    prior : str
        Type of prior used, if uniform prime prior is enabled.
    """
    def __init__(self, parameters=None, prior_bounds=None, prior=None,
                 rescale_bounds=None, boundary_inversion=[],
                 inversion_type='split', detect_edges_kwargs={}):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        if rescale_bounds is None:
            logger.info('Using default bounds: [-1, 1]')
            self.rescale_bounds = {p: [-1, 1] for p in self.parameters}
        else:
            if isinstance(rescale_bounds, list):
                self.rescale_bounds = \
                    {p: rescale_bounds for p in self.parameters}
            elif isinstance(rescale_bounds, dict):
                if s := set(parameters) - set(rescale_bounds.keys()):
                    raise RuntimeError(f'Missing bounds for parameters {s}')
                self.rescale_bounds = rescale_bounds
            else:
                raise TypeError(
                    'rescale_bounds must be an instance of list or dict. '
                    f'Got type: {type(rescale_bounds).__name__}')

        self._rescale_factor = \
            {p: np.ptp(self.rescale_bounds[p]) for p in self.parameters}
        self._rescale_shift = \
            {p: self.rescale_bounds[p][0] for p in self.parameters}

        if isinstance(boundary_inversion, list):
            self.boundary_inversion = \
                {n: inversion_type for n in boundary_inversion}
        elif isinstance(boundary_inversion, dict):
            self.boundary_inversion = boundary_inversion
        else:
            raise TypeError(
                'boundary_inversion must be an instance of list or dict. '
                f'Got type: {type(boundary_inversion).__name__}')

        for p in self.boundary_inversion:
            self.rescale_bounds[p] = [0, 1]

        self._edges = {n: None for n in self.boundary_inversion}
        self.detect_edges_kwargs = detect_edges_kwargs

        if prior is not None:
            raise NotImplementedError

        self.update_bounds(self.prior_bounds)

    def _rescale_to_bounds(self, x, n):
        out = self._rescale_factor[n] * \
                ((x - self.bounds[n][0]) /
                 (self.bounds[n][1] - self.bounds[n][0])) \
                + self._rescale_shift[n]

        log_j = (-np.log(self.bounds[n][1] - self.bounds[n][0])
                 + np.log(self._rescale_factor[n]))
        return out, log_j

    def _inverse_rescale_to_bounds(self, x, n):
        out = (self.bounds[n][1] - self.bounds[n][0]) \
               * (x - self._rescale_shift[n]) \
               / self._rescale_factor[n] + self.bounds[n][0]

        log_j = (np.log(self.bounds[n][1] - self.bounds[n][0])
                 - np.log(self._rescale_factor[n]))

        return out, log_j

    def _apply_inversion(self, x, x_prime, log_j, p, pp, compute_radius,
                         test=None):
        if self._edges[p] is None:
            self._edges[p] = detect_edge(x[p], test=test,
                                         **self.detect_edges_kwargs)

        if self._edges[p]:
            logger.debug(f'Apply inversion for {p} to {self._edges[p]} bound')
            logger.debug('Fixing bounds to [0, 1]')
            logger.debug('Rescaling')
            x_prime[pp], lj = rescale_zero_to_one(x[p], *self.bounds[p])
            log_j += lj
            if self._edges[p] == 'upper':
                x_prime[pp] = 1 - x_prime[pp]
            if (self.boundary_inversion[p] == 'duplicate' or compute_radius):
                logger.debug('Inverting with duplication')
                x_inv = x_prime.copy()
                x_inv[pp] *= -1
                x_prime = np.concatenate([x_prime, x_inv])
                x = np.concatenate([x,  x])
                log_j = np.concatenate([log_j, log_j])
            else:
                logger.debug('Inverting with splitting')
                inv = np.random.choice(x_prime.size,
                                       x_prime.size // 2,
                                       replace=False)
                x_prime[pp][inv] *= -1
        else:
            logger.debug(f'Not using inversion for {p}')
            logger.debug('Rescaling to [-1, 1]')
            x_prime[pp], lj = self._rescale_to_bounds(x[p], p)
            log_j += lj

        return x, x_prime, log_j

    def _reverse_inversion(self, x, x_prime, log_j, p, pp):
        if self._edges[p]:
            inv = x_prime[pp] < 0.
            x_prime[pp][~inv] = x_prime[pp][~inv]
            x_prime[pp][inv] = -x_prime[pp][inv]
            if self._edges[p] == 'upper':
                x_prime[pp] = 1 - x_prime[pp]
            x[p], lj = inverse_rescale_zero_to_one(
                x_prime[pp], *self.bounds[p])
            log_j += lj
        else:
            x[p], lj = self._inverse_rescale_to_bounds(x_prime[pp], p)
            log_j += lj
        return x, x_prime, log_j

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
            if p in self.boundary_inversion:
                x, x_prime, log_j = self._apply_inversion(
                        x, x_prime, log_j, p, pp, compute_radius, **kwargs)
            else:
                x_prime[pp], lj = self._rescale_to_bounds(x[p], p)
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
                log_j += lj
        return x, x_prime, log_j

    def update_bounds(self, x):
        """Update the bounds used for the reparameterisation"""
        self.bounds = \
            {p: [np.min(x[p]), np.max(x[p])] for p in self.parameters}
        self.prime_prior_bounds = \
            {pp: self._rescale_to_bounds(np.asarray(self.bounds[p])[0], p)
             for p, pp in zip(self.parameters, self.prime_parameters)}

        logger.debug(f'New bounds: {self.bounds}')

    def reset_inversion(self):
        """Reset the edges for inversion"""
        self._edges = {n: None for n in self.boundary_inversion}

    def x_prime_log_prior(self, x_prime):
        """Compute the prior in the prime space assuming a uniform prior"""
        if self.prime_prior:
            log_p = 0
            for pp in self.prime_parameters:
                # Do something here
                pass
            return log_p
        else:
            return None


class Angle(Reparameterisation):
    """Reparameterisation for a single angle"""
    def __init__(self, parameters=None, bounds=None, radial=None, scale=1.0,
                 prior=None):
        if len(parameters) == 1:
            parameters.append(parameters[0] + '_radial')
            self.chi = stats.chi(2)
        else:
            self.chi = False

        self.parameters = parameters

        self.scale = scale
        self.bounds = bounds

        if bounds[0] == 0:
            self._zero_bound = True
        else:
            self._zero_bound = False

        self.prime_parameters = [self.angle + '_x', self.angle + '_y']
        self.requires = []

    @property
    def angle(self):
        return self.parameters[0]

    @property
    def radial(self):
        return self.parameters[1]

    @property
    def x(self):
        return self.prime_parmeters[0]

    @property
    def y(self):
        return self.prime_parmeters[1]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert the angle to Cartesian coordinates"""
        if self.chi:
            r = self.chi.rvs(size=x.size)
        else:
            r = x[self.radial]
        if any(r < 0):
            raise RuntimeError('Radius cannot be negative.')
        x_prime[self.prime_parameters[0]] = \
            r * np.cos(self.scale * x[self.angle])
        x_prime[self.prime_parameters[1]] = \
            r * np.sin(self.scale * x[self.angle])
        log_j += np.log(r)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to an angle"""

        x[self.radius] = np.sqrt(x_prime[self.x] ** 2 + x_prime[self.y] ** 2)
        if self._zero_bound:
            x[self.angle] = \
                np.arctan2(x_prime[self.y], x_prime[self.x]) % (2. * np.pi) / \
                self.scale
        else:
            x[self.angle] = \
                np.arctan2(x_prime[self.y], x_prime[self.x]) / self.scale

        log_j -= np.log(x[self.radius])

        return x, x_prime, log_j


REPARAMETERISATION_DICT = {
    'default': RescaleToBounds,
    'rescaletobounds': RescaleToBounds
}
