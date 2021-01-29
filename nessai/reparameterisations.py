import logging

import numpy as np
from scipy import stats

from .utils import (
    detect_edge,
    configure_edge_detection,
    rescale_zero_to_one,
    inverse_rescale_zero_to_one
)


logger = logging.getLogger(__name__)


def get_reparameterisation(reparameterisation):
    """Function to get a reparmeterisation class from a name"""
    rc, kwargs = default_reparameterisations.get(reparameterisation, None)
    if rc is None:
        raise ValueError(f'Unknown reparameterisation: {reparameterisation}')
    else:
        return rc, kwargs


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

        if isinstance(prior_bounds, (list, tuple)):
            if len(prior_bounds) == 2:
                prior_bounds = {self.parameters[0]: np.asarray(prior_bounds)}
            else:
                raise RuntimeError('Prior bounds got a list of len > 2')
        elif not isinstance(prior_bounds, dict):
            raise TypeError('Prior bounds must be dict or tuple of len 2')

        if set(self.parameters) - set(prior_bounds.keys()):
            raise RuntimeError(
                'Mismatch between parameters and prior bounds: '
                f'{set(self.parameters)}, {set(prior_bounds.keys())}')

        self.prior_bounds = {p: np.asarray(b) for p, b in prior_bounds.items()}
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

    def add_reparameterisation(self, reparameterisation):
        """Add a reparameterisation"""
        self.add_reparameterisations(reparameterisation)

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
    """Reparameteristion that does not modify the parameters"""
    def __init__(self, parameters=None):
        if not isinstance(parameters, (str, list)):
            raise TypeError('Parameters must be a str or list.')

        self.parameters = \
            [parameters] if isinstance(parameters, str) else parameters

        self.prime_parameters = self.parameters
        self.requires = []
        logger.debug(f'Initialised reparameterisation: {self.name}')

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
        x[self.parameters] = x_prime[self.prime_parameters]
        return x, x_prime, log_j


class RescaleToBounds(Reparameterisation):
    """Reparmeterisation that maps to the specified interval.

    By default the interval is [-1, 1]. Also includes options for
    boundary inversion.

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
                 detect_edges=False, inversion_type='split',
                 detect_edges_kwargs={}, offset=False):

        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        if rescale_bounds is None:
            logger.debug('Using default rescale bounds: [-1, 1]')
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

        if boundary_inversion:
            if isinstance(boundary_inversion, list):
                self.boundary_inversion = \
                    {n: inversion_type for n in boundary_inversion}
            elif isinstance(boundary_inversion, dict):
                self.boundary_inversion = boundary_inversion
            elif isinstance(boundary_inversion, bool):
                self.boundary_inversion = \
                    {p: inversion_type for p in self.parameters}
            else:
                raise TypeError(
                    'boundary_inversion must be an instance of list or dict. '
                    f'Got type: {type(boundary_inversion).__name__}')
        else:
            self.boundary_inversion = []

        for p in self.boundary_inversion:
            self.rescale_bounds[p] = [0, 1]

        self._edges = {n: None for n in self.boundary_inversion}
        self.detect_edges = detect_edges
        self.detect_edges_kwargs = \
            configure_edge_detection(detect_edges_kwargs, detect_edges)

        if self.detect_edges and not self.boundary_inversion:
            raise RuntimeError(
                'Must enable boundary inversion to detect edges')

        if prior is not None:
            raise NotImplementedError

        if offset:
            self.offsets = {p: b[0] + np.ptp(b) / 2
                            for p, b in self.prior_bounds.items()}
        else:
            self.offsets = {p: 0. for p in self.prior_bounds.keys()}

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
                         test=None, detect_edge_prime=False):
        if self._edges[p] is None:
            if detect_edge_prime:
                self._edges[p] = detect_edge(x_prime[pp], test=test,
                                             **self.detect_edges_kwargs)
            else:
                self._edges[p] = detect_edge(x[p], test=test,
                                             **self.detect_edges_kwargs)

        if self._edges[p]:
            logger.debug(f'Apply inversion for {p} to {self._edges[p]} bound')
            logger.debug('Fixing bounds to [0, 1]')
            logger.debug('Rescaling')
            x_prime[pp], lj = \
                rescale_zero_to_one(x[p] - self.offsets[p], *self.bounds[p])
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
            x_prime[pp], lj = \
                self._rescale_to_bounds(x[p] - self.offsets[p], p)
            log_j += lj

        return x, x_prime, log_j

    def _reverse_inversion(self, x, x_prime, log_j, p, pp):
        if self._edges[p]:
            inv = x_prime[pp] < 0.
            x[p][~inv] = x_prime[pp][~inv]
            x[p][inv] = -x_prime[pp][inv]
            if self._edges[p] == 'upper':
                x[p] = 1 - x[p]
            x[p], lj = inverse_rescale_zero_to_one(
                x[p], *self.bounds[p])
            log_j += lj
        else:
            x[p], lj = self._inverse_rescale_to_bounds(x_prime[pp], p)
            x[p] += self.offsets[p]
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
                x_prime[pp], lj = \
                    self._rescale_to_bounds(x[p] - self.offsets[p], p)
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
        return x, x_prime, log_j

    def update_bounds(self, x):
        """Update the bounds used for the reparameterisation"""
        self.bounds = \
            {p: [np.min(x[p] - self.offsets[p]),
                 np.max(x[p] - self.offsets[p])]
             for p in self.parameters}
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
    def __init__(self, parameters=None, prior_bounds=None, radial=None,
                 scale=1.0, prior=None):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        if len(self.parameters) == 1:
            self.parameters.append(self.parameters[0] + '_radial')
            self.chi = stats.chi(2)
        elif len(self.parameters) == 2:
            self.chi = False
        else:
            raise RuntimeError

        self.scale = scale

        if prior_bounds[self.angle][0] == 0:
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
    def radius(self):
        return self.parameters[1]

    @property
    def x(self):
        return self.prime_parameters[0]

    @property
    def y(self):
        return self.prime_parameters[1]

    def _rescale_radius(self, x, x_prime, log_j, **kwargs):
        return x[self.radial], x, x_prime, log_j

    def _rescale_angle(self, x, x_prime, log_j, **kwargs):
        return x[self.angle] * self.scale, x, x_prime, log_j

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        return x, x_prime, log_j

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert the angle to Cartesian coordinates"""
        if self.chi:
            r = self.chi.rvs(size=x.size)
        else:
            r, x, x_prime, log_j = self._rescale_radial(
                x, x_prime, log_j, **kwargs)
        if any(r < 0):
            raise RuntimeError('Radius cannot be negative.')

        angle, x, x_prime, log_j = self._rescale_angle(
            x, x_prime, log_j, **kwargs)

        x_prime[self.prime_parameters[0]] = r * np.cos(angle)
        x_prime[self.prime_parameters[1]] = r * np.sin(angle)
        log_j += np.log(r)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to an angle"""
        x[self.radial] = np.sqrt(x_prime[self.x] ** 2 + x_prime[self.y] ** 2)
        if self._zero_bound:
            x[self.angle] = \
                np.arctan2(x_prime[self.y], x_prime[self.x]) % (2. * np.pi) / \
                self.scale
        else:
            x[self.angle] = \
                np.arctan2(x_prime[self.y], x_prime[self.x]) / self.scale

        log_j -= np.log(x[self.radial])
        x, x_prime, log_j = self._inverse_rescale_angle(x, x_prime, log_j)

        return x, x_prime, log_j


class ToCartesian(Angle):
    """Convert a paraemter to Cartesian coordinates"""
    def __init__(self, mode='split', scale=np.pi, **kwargs):
        super().__init__(scale=np.pi, **kwargs)

        self.mode = mode
        if self.mode not in ['duplicate', 'split', 'half']:
            raise RuntimeError(f'Unknown mode: {self.mode}')
        logger.debug(f'Using mode: {self.mode}')

        self._zero_bound = False

    def _rescale_angle(self, x, x_prime, log_j, compute_radius=False,
                       **kwargs):
        angle, lj = rescale_zero_to_one(
            x[self.angle], *self.prior_bounds[self.angle])
        log_j += lj
        if self.mode == 'duplicate' or compute_radius:
            angle = np.concatenate([-angle, angle])
            x = np.concatenate([x, x])
            x_prime = np.concatenate([x_prime, x_prime])
        elif self.mode == 'split':
            neg = np.random.choice(angle.size, angle.size // 2, replace=False)
            angle[neg] *= -1
        elif self.mode == 'half':
            pass
        angle *= self.scale
        return angle, x, x_prime, log_j

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        x[self.angle], lj = inverse_rescale_zero_to_one(
            np.abs(x[self.angle]), *self.prior_bounds[self.angle])
        log_j += lj
        return x, x_prime, log_j


class AnglePair(Reparameterisation):
    """Reparameterisation for a pair of angles and a radial component"""
    def __init__(self, parameters=None, prior_bounds=None, radial=None,
                 prior=None, convention=None):

        self._conventions = ['az-zen', 'ra-dec']

        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        # Determine which parameter is for the horizontal plane
        # and which is for the vertical plane
        logger.debug('Checking order of parameters')
        b = np.ptp([prior_bounds[p] for p in parameters], axis=1)
        hz = np.where(b == (2 * np.pi))[0][0]
        vt = np.where(b == np.pi)[0][0]
        parameters[0], parameters[1] = parameters[hz], parameters[vt]

        if len(parameters) == 2:
            m = '_'.join(parameters)
            parameters.append(f'{m}_radial')
            self.chi = stats.chi(3)
        elif len(parameters) not in [2, 3]:
            raise RuntimeError(
                'Must use a pair of angles or a pair plus a radius')
        else:
            m = '_'.join(parameters)
            self.chi = False

        logger.debug(f'Parameters are: {parameters}')

        self.parameters = parameters
        self.prime_parameters = [f'{m}_{x}' for x in ['x', 'y', 'z']]

        if convention is None:
            if (self.prior_bounds[self.parameters[1]][0] == 0 and
                    self.prior_bounds[self.parameters[1]][1] == np.pi):
                self.convention = 'az-sen'
            elif (self.prior_bounds[self.parameters[1]][0] == -np.pi / 2 and
                    self.prior_bounds[self.parameters[1]][1] == np.pi / 2):
                self.convention = 'ra-dec'
            else:
                raise RuntimeError
        elif convention in self._conventions:
            self.convention = convention
        else:
            raise RuntimeError(f'Unknown convention: {convention}')

        logger.debug(f'Using convention: {self.convention}')

        self.requires = []

    @property
    def angles(self):
        return self.parameters[:1]

    @property
    def radial(self):
        return self.parameters[-1]

    @property
    def x(self):
        return self.prime_parameters[0]

    @property
    def y(self):
        return self.prime_parameters[1]

    @property
    def z(self):
        return self.prime_parameters[2]

    def _az_zen(self, x, x_prime, log_j, r):
        x_prime[self.prime_parameters[0]] = \
            r * np.sin(x[self.parameters[1]]) * np.cos(x[self.parameters[0]])
        x_prime[self.prime_parameters[1]] = \
            r * np.sin(x[self.parametesr[1]]) * np.sin(x[self.parameters[0]])
        x_prime[self.prime_parameters[2]] = \
            r * np.cos(x[self.parameters[1]])
        log_j += (2 * np.log(r) + np.log(np.sin(x[self.parameters[1]])))
        return x, x_prime, log_j

    def _ra_dec(self, x, x_prime, log_j, r):
        x_prime[self.prime_parameters[0]] = \
            r * np.cos(x[self.parameters[1]]) * np.cos(x[self.parameters[0]])
        x_prime[self.prime_parameters[1]] = \
            r * np.cos(x[self.parameters[1]]) * np.sin(x[self.parameters[0]])
        x_prime[self.prime_parameters[2]] = \
            r * np.sin(x[self.parameters[1]])
        log_j += (2 * np.log(r) + np.log(np.cos(x[self.parameters[1]])))
        return x, x_prime, log_j

    def _inv_az_zen(self, x, x_prime, log_j):
        x[self.radial] = np.sqrt(x_prime[self.x] ** 2. + x_prime[self.y] ** 2.
                                 + x_prime[self.z] ** 2.)
        x[self.parameters[0]] = \
            np.arctan2(x_prime[self.y], x_prime[self.x]) % (2. * np.pi)
        x[self.parameters[1]] = \
            np.arctan2(np.sqrt(x_prime[self.x] ** 2. + x_prime[self.y] ** 2.0),
                       x_prime[self.z])
        log_j += (-2 * np.log(x[self.radial])
                  - np.log(np.sin(x[self.parameters[1]])))

        return x, x_prime, log_j

    def _inv_ra_dec(self, x, x_prime, log_j):
        x[self.radial] = np.sqrt(x_prime[self.x] ** 2. + x_prime[self.y] ** 2.
                                 + x_prime[self.z] ** 2.)
        x[self.parameters[0]] = \
            np.arctan2(x_prime[self.y], x_prime[self.x]) % (2. * np.pi)
        x[self.parameters[1]] = \
            np.arctan2(x_prime[self.z],
                       np.sqrt(x_prime[self.x] ** 2. + x_prime[self.y] ** 2.0))
        log_j += (-2 * np.log(x[self.radial])
                  - np.log(np.cos(x[self.parameters[1]])))
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
            raise RuntimeError('Radius cannot be negative.')

        if self.convention == 'az-zen':
            return self._az_zen(x, x_prime, log_j, r)
        else:
            return self._ra_dec(x, x_prime, log_j, r)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to spherical polar angles"""
        if self.convention == 'az-zen':
            return self._inv_az_zen(x, x_prime, log_j)
        else:
            return self._inv_ra_dec(x, x_prime, log_j)
        return x, x_prime, log_j

    def log_prime_prior(self, x_prime):
        """
        Log probability of 3d Cartesian coordinates for an isotropic
        distribution of angles and a radial component drawn from a chi
        distribution with three degrees of freedom.
        """
        if not self.chi:
            raise RuntimeError('Prime prior not define with radial component')
        return - 1.5 * np.log(2 * np.pi) \
               - np.sum(x_prime[self.prime_parameters] ** 2, axis=0) / 2


default_reparameterisations = {
    'default': (RescaleToBounds, {}),
    'rescaletobounds': (RescaleToBounds, {}),
    'inversion': (RescaleToBounds, {'detect_edges': True,
                                    'boundary_inversion': True,
                                    'inversion_type': 'split'}),
    'inversion-duplicate': (RescaleToBounds, {'detect_edges': True,
                                              'boundary_inversion': True,
                                              'inversion_type': 'duplicate'}),
    'angle-pi': (Angle, {'scale': 2.0, 'prior': 'uniform'}),
    'angle-2pi': (Angle, {'scale': 1.0, 'prior': 'uniform'}),
    'angle-sine': (Angle, {'scale': 1.0, 'prior': 'sine'}),
    'to-cartesian': (ToCartesian, {}),
    'none': (NullReparameterisation, {}),
    None: (NullReparameterisation, {}),
}
