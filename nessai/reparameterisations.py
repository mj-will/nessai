# -*- coding: utf-8 -*-
"""
Functions and objects related to reparametersiations for use in the
``reparameterisations`` dictionary.

See the documentation for an in-depth description of how to use these
functions and classes.
"""
import logging

import numpy as np
from scipy import stats

from .utils import (
    detect_edge,
    configure_edge_detection,
    rescale_zero_to_one,
    inverse_rescale_zero_to_one,
    rescale_minus_one_to_one,
    inverse_rescale_minus_one_to_one,
    determine_rescaled_bounds,
    rescaling_functions
)

from .priors import (
    log_uniform_prior,
    log_2d_cartesian_prior,
    log_2d_cartesian_prior_sine,
    )


logger = logging.getLogger(__name__)


def get_reparameterisation(reparameterisation):
    """Function to get a reparmeterisation class from a name

    Parameters
    ----------
    reparameterisation : str, \
            :obj:`nessai.reparameterisations.Reparametersiation`
        Name of the reparameterisations to return or a class that inherits from
        :obj:`~nessai.reparameterisations.Reparmeterisation`

    Returns
    -------
    :obj:`nessai.reparameteristaions.Reparmeterisation`
        The class to use for the reparameterisation. The class is NOT
        initialised.
    dict
        Keyword arguments for the specific reparameterisations.
    """
    if isinstance(reparameterisation, str):
        rc, kwargs = default_reparameterisations.get(
            reparameterisation, (None, None))
        if rc is None:
            raise ValueError(
                f'Unknown reparameterisation: {reparameterisation}')
        else:
            if kwargs is None:
                kwargs = {}
            else:
                kwargs = kwargs.copy()
            return rc, kwargs

    elif (isinstance(reparameterisation, type) and
            issubclass(reparameterisation, Reparameterisation)):
        return reparameterisation, {}
    elif reparameterisation is None:
        return NullReparameterisation, {}
    else:
        raise RuntimeError('Reparameterisation must be a str, None, or '
                           'a class that inherits from `Reparameterisation`')


class Reparameterisation:
    """
    Base object for reparameterisations.

    Parameters
    ----------
    parameters : str or list
        Name of parameters to reparameterise.
    """
    _update_bounds = False
    has_prior = False
    has_prime_prior = False
    requires_prime_prior = False

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


class CombinedReparameterisation(dict):
    """Class to handle mulitple reparameterisations

    Parameters
    ----------
    reparameterisations : list, optional
        List of reparameterisations to add to the combined reparameterisations.
        Further reparameterisations can be added using
        `add_reparameterisations`.
    """
    def __init__(self, reparameterisations=None):
        super().__init__()
        self.reparameterisations = {}
        self.parameters = []
        self.prime_parameters = []
        self.requires = []
        if reparameterisations is not None:
            self.add_reparameterisations(reparameterisations)

    @property
    def has_prime_prior(self):
        """Boolean to check if prime prior can be enabled"""
        return all(r.has_prime_prior for r in self.values())

    @property
    def requires_prime_prior(self):
        """Boolean to check if any of the priors require the prime space"""
        return any(r.requires_prime_prior for r in self.values())

    def _add_reparameterisation(self, reparameterisation):
        requires = reparameterisation.requires
        if (requires and
                (requires not in self.parameters or
                    requires not in self.prime_parameters)):
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
        log_j : array_like
            Log jacobian to be updated
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
        log_j : array_like
            Log jacobian to be updated
        """
        for r in reversed(list(self.values())):
            x, x_prime, log_j = r.inverse_reparameterise(
                x, x_prime, log_j, **kwargs)
        return x, x_prime, log_j

    def update_bounds(self, x):
        """
        Update the bounds used for the reparameterisation
        """
        for r in self.values():
            if hasattr(r, 'update_bounds'):
                logger.debug(f'Updating bounds for: {r.name}')
                r.update_bounds(x)

    def reset_inversion(self):
        """
        Reset edges for boundary inversion
        """
        for r in self.values():
            if hasattr(r, 'reset_inversion'):
                r.reset_inversion()

    def log_prior(self, x):
        """
        Compute any additional priors for auxiliary parameters
        """
        log_p = 0
        for r in self.values():
            if r.has_prior:
                log_p += r.log_prior(x)
        return log_p

    def x_prime_log_prior(self, x_prime):
        """
        Compute the prior in the prime space
        """
        log_p = 0
        for r in self.values():
            log_p += r.x_prime_log_prior(x_prime)
        return log_p


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


class Rescale(Reparameterisation):
    """Reparameterisation that rescales the parameters by a constant factor
    that does not depend on the prior bounds.

    Parameters
    ----------
    scale : float
        Scaling constant.
    """
    def __init__(self, parameters=None, scale=None, prior_bounds=None):
        if scale is None:
            raise RuntimeError('Must specify a scale!')
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        if isinstance(scale, (int, float)):
            self.scale = {p: float(scale) for p in self.parameters}
        elif isinstance(scale, list):
            if not len(scale) == len(self.parameters):
                raise RuntimeError(
                    'Scale list is a different length to the parameters.')
            self.scale = {p: float(s) for p, s in zip(self.parameters, scale)}
        elif isinstance(scale, dict):
            if not set(self.parameters) == set(scale.keys()):
                raise RuntimeError('Mismatched parameters with scale dict')
            scale = {p: float(s) for p, s in scale.items()}
            self.scale = scale
        else:
            raise TypeError(
                'Scale input must be an instance of int, list or dict')

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
            x[p] = x_prime[pp] * self.scale[p]
            log_j += np.log(np.abs(self.scale[p]))
        return x, x_prime, log_j


class RescaleToBounds(Reparameterisation):
    """Reparmeterisation that maps to the specified interval.

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
        applied.
    """
    def __init__(self, parameters=None, prior_bounds=None, prior=None,
                 rescale_bounds=None, boundary_inversion=None,
                 detect_edges=False, inversion_type='split',
                 detect_edges_kwargs=None, offset=False, update_bounds=True,
                 pre_rescaling=None, post_rescaling=None):

        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        self.bounds = None
        self.detect_edge_prime = False

        self.has_pre_rescaling = True
        self.has_post_rescaling = True

        if rescale_bounds is None:
            logger.debug('Using default rescale bounds: [-1, 1]')
            self.rescale_bounds = {p: [-1, 1] for p in self.parameters}
        else:
            if isinstance(rescale_bounds, list):
                self.rescale_bounds = \
                    {p: rescale_bounds for p in self.parameters}
            elif isinstance(rescale_bounds, dict):
                s = set(parameters) - set(rescale_bounds.keys())
                if s:
                    raise RuntimeError(f'Missing bounds for parameters {s}')
                self.rescale_bounds = rescale_bounds
            else:
                raise TypeError(
                    'rescale_bounds must be an instance of list or dict. '
                    f'Got type: {type(rescale_bounds).__name__}')

        if boundary_inversion is not None:
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

        self._update_bounds = update_bounds if not detect_edges else True
        self._edges = {n: None for n in self.parameters}
        self.detect_edges = detect_edges
        if self.boundary_inversion:
            self.detect_edges_kwargs = \
                configure_edge_detection(detect_edges_kwargs,
                                         self.detect_edges)

        if self.detect_edges and not self.boundary_inversion:
            raise RuntimeError(
                'Must enable boundary inversion to detect edges')

        if prior == 'uniform':
            self.prior = 'uniform'
            self.has_prime_prior = True
            self._prime_prior = log_uniform_prior
            logger.info(f'Prime prior enabled for {self.name}')
        else:
            self.has_prime_prior = False
            logger.info(f'Prime prior disabled for {self.name}')

        if offset:
            self.offsets = {p: b[0] + np.ptp(b) / 2
                            for p, b in self.prior_bounds.items()}
            logger.debug(f'Offsets: {self.offsets}')
        else:
            self.offsets = {p: 0. for p in self.prior_bounds.keys()}

        self.configure_pre_rescaling(pre_rescaling)
        self.configure_post_rescaling(post_rescaling)

        self.set_bounds(self.prior_bounds)

    def configure_pre_rescaling(self, pre_rescaling):
        if pre_rescaling is not None:
            if isinstance(pre_rescaling, str):
                logger.debug(f'Getting pre-rescaling function {pre_rescaling}')
                self.pre_rescaling, self.pre_rescaling_inv = \
                    rescaling_functions.get(
                        pre_rescaling.lower(), (None, None))
                if self.pre_rescaling is None:
                    raise RuntimeError(
                        f'Unkown rescaling function: {pre_rescaling}')
            elif len(pre_rescaling) == 2:
                self.pre_rescaling = pre_rescaling[0]
                self.pre_rescaling_inv = pre_rescaling[1]
            else:
                raise RuntimeError(
                    'Pre-rescaling must be str or tuple of two functions')
            logger.debug('Disabling prime prior with pre-rescaling')
            self.has_prior_prior = False
            self.has_pre_rescaling = True
        else:
            logger.debug('No pre-rescaling to configure')
            self.has_pre_rescaling = False

    def configure_post_rescaling(self, post_rescaling):

        if post_rescaling is not None:
            if isinstance(post_rescaling, str):
                logger.debug(
                    f'Getting post-rescaling function {post_rescaling}')
                self.post_rescaling, self.post_rescaling_inv = \
                    rescaling_functions.get(
                        post_rescaling.lower(), (None, None))
                if self.post_rescaling is None:
                    raise RuntimeError(
                        f'Unkown rescaling function: {post_rescaling}')
            elif len(post_rescaling) == 2:
                self.post_rescaling = post_rescaling[0]
                self.post_rescaling_inv = post_rescaling[1]
            else:
                raise RuntimeError(
                    'Post-rescaling must be str or tuple of two functions')
            logger.debug('Disabling prime prior with post-rescaling')
            self.has_prior_prior = False

            if post_rescaling == 'logit':
                if self._update_bounds:
                    raise RuntimeError('Cannot use logit with update bounds')
                logger.debug('Setting bounds to [0, 1] for logit')
                self.rescale_bounds = {p: [0, 1] for p in self.parameters}
            self.has_post_rescaling = True
        else:
            logger.debug('No post-rescaling to configure')
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
            self._edges[p] = detect_edge(x_prime[pp], test=test,
                                         **self.detect_edges_kwargs)
            self.update_prime_prior_bounds()

        if self._edges[p]:
            logger.debug(f'Apply inversion for {p} to {self._edges[p]} bound')
            logger.debug('Fixing bounds to [0, 1]')
            logger.debug('Rescaling')
            x_prime[pp], lj = \
                rescale_zero_to_one(x_prime[pp] - self.offsets[p],
                                    *self.bounds[p])
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
            logger.debug(f'Rescaling to {self.rescale_bounds[p]}')
            x_prime[pp], lj = \
                rescale_minus_one_to_one(x_prime[pp] - self.offsets[p],
                                         xmin=self.bounds[p][0],
                                         xmax=self.bounds[p][1])

            log_j += lj

        return x, x_prime, log_j

    def _reverse_inversion(self, x, x_prime, log_j, p, pp):
        if self._edges[p]:
            inv = x[p] < 0.
            x[p][~inv] = x[p][~inv]
            x[p][inv] = -x[p][inv]

            if self._edges[p] == 'upper':
                x[p] = 1 - x[p]
            x[p], lj = inverse_rescale_zero_to_one(
                x[p], *self.bounds[p])
            x[p] += self.offsets[p]
            log_j += lj
        else:
            x[p], lj = inverse_rescale_minus_one_to_one(
                x[p], xmin=self.bounds[p][0], xmax=self.bounds[p][1])
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
            if self.has_pre_rescaling:
                x_prime[pp], lj = self.pre_rescaling(x[p])
                log_j += lj
            else:
                x_prime[pp] = x[p].copy()

            if p in self.boundary_inversion:
                x, x_prime, log_j = self._apply_inversion(
                        x, x_prime, log_j, p, pp, compute_radius, **kwargs)
            else:
                x_prime[pp], lj = \
                    self._rescale_to_bounds(x[p] - self.offsets[p], p)
                log_j += lj
            if self.has_post_rescaling:
                x_prime[pp], lj = self.post_rescaling(x_prime[pp])
                log_j += lj
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Map inputs to the physical space from the prime space"""
        for p, pp in zip(reversed(self.parameters),
                         reversed(self.prime_parameters)):
            if self.has_post_rescaling:
                x[p], lj = self.post_rescaling_inv(x_prime[pp])
                log_j += lj
            else:
                x[p] = x_prime[pp].copy()
            if p in self.boundary_inversion:
                x, x_prime, log_j = self._reverse_inversion(
                    x, x_prime, log_j, p, pp, **kwargs)
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
        self._edges = {n: None for n in self.parameters}

    def set_bounds(self, prior_bounds):
        """Set the initial bounds for rescaling"""
        self._rescale_factor = \
            {p: np.ptp(self.rescale_bounds[p]) for p in self.parameters}
        self._rescale_shift = \
            {p: self.rescale_bounds[p][0] for p in self.parameters}

        self.prior_bounds = \
            {p: self.pre_rescaling(prior_bounds[p])[0]
             for p in self.parameters}
        self.bounds = {p: self.pre_rescaling(b - self.offsets[p])[0]
                       for p, b in prior_bounds.items()}
        logger.debug(f'Initial bounds: {self.bounds}')
        self.update_prime_prior_bounds()

    def update_bounds(self, x):
        """Update the bounds used for the reparameterisation"""
        if self._update_bounds:
            self.bounds = \
                {p: [self.pre_rescaling(np.min(x[p]))[0] - self.offsets[p],
                     self.pre_rescaling(np.max(x[p]))[0] - self.offsets[p]]
                 for p in self.parameters}
            logger.debug(f'New bounds: {self.bounds}')
            self.update_prime_prior_bounds()

        else:
            logger.debug('Update bounds not enabled')

    def update_prime_prior_bounds(self):
        """Update the prior bounds used for the prime prior"""
        if self.has_prime_prior:
            self.prime_prior_bounds = \
                {pp: self.post_rescaling(np.asarray(determine_rescaled_bounds(
                    self.prior_bounds[p][0], self.prior_bounds[p][1],
                    self.bounds[p][0], self.bounds[p][1], self._edges[p],
                    self.offsets[p], rescale_bounds=self.rescale_bounds[p],
                    inversion=p in self.boundary_inversion)))[0]
                 for p, pp in zip(self.parameters, self.prime_parameters)}
            logger.debug(f'New prime bounds: {self.prime_prior_bounds}')

    def x_prime_log_prior(self, x_prime):
        """Compute the prior in the prime space assuming a uniform prior"""
        if self.has_prime_prior:
            log_p = 0
            for pp in self.prime_parameters:
                log_p += self._prime_prior(x_prime[pp],
                                           *self.prime_prior_bounds[pp])
            return log_p
        else:
            raise RuntimeError(
                f'Prime prior is not configured for {self.name}')


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
        coordiantes.
    prior : {'uniform', 'sine', None}
        Type of prior being used for sampling this angle. If specified, the
        prime prior is enabled. If None then it is disabled.
    """
    def __init__(self, parameters=None, prior_bounds=None, scale=1.0,
                 prior=None):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        if len(self.parameters) == 1:
            self.parameters.append(self.parameters[0] + '_radial')
            self.chi = stats.chi(2)
            self.has_prior = True
        elif len(self.parameters) == 2:
            self.chi = False
        else:
            raise RuntimeError('Too many parameters for Angle')

        self.scale = scale

        if prior_bounds[self.angle][0] == 0:
            self._zero_bound = True
        else:
            self._zero_bound = False

        self.prime_parameters = [self.angle + '_x', self.angle + '_y']
        self.requires = []

        if prior in ['uniform', 'sine']:
            self.prior = prior
            self.has_prime_prior = True
            if self.prior == 'uniform':
                self._prime_prior = log_2d_cartesian_prior
                self._k = self.scale * np.pi
            else:
                self._prime_prior = log_2d_cartesian_prior_sine
                self._k = np.pi
            logger.info(f'Prime prior enabled for {self.name}')
        else:
            self.has_prime_prior = False
            logger.info(f'Prime prior disabled for {self.name}')

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
            x, x_prime, log_j, **kwargs)

        if self.chi:
            r = self.chi.rvs(size=x.size)
        else:
            r, x, x_prime, log_j = self._rescale_radial(
                x, x_prime, log_j, **kwargs)
        if any(r < 0):
            raise RuntimeError('Radius cannot be negative.')

        x_prime[self.prime_parameters[0]] = r * np.cos(angle)
        x_prime[self.prime_parameters[1]] = r * np.sin(angle)
        log_j += np.log(r)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from Cartesian to an angle and radial component"""
        x[self.parameters[1]] = \
            np.sqrt(x_prime[self.prime_parameters[0]] ** 2
                    + x_prime[self.prime_parameters[1]] ** 2)
        if self._zero_bound:
            x[self.parameters[0]] = \
                np.arctan2(x_prime[self.prime_parameters[1]],
                           x_prime[self.prime_parameters[0]]) % \
                (2. * np.pi) / self.scale
        else:
            x[self.parameters[0]] = \
                np.arctan2(x_prime[self.prime_parameters[1]],
                           x_prime[self.prime_parameters[0]]) / self.scale

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
                k=self._k)
        else:
            raise RuntimeError('Prime prior')


class ToCartesian(Angle):
    """Convert a paraemter to Cartesian coordinates"""
    def __init__(self, mode='split', scale=np.pi, **kwargs):
        super().__init__(scale=np.pi, **kwargs)

        self.mode = mode
        if self.mode not in ['duplicate', 'split', 'half']:
            raise RuntimeError(f'Unknown mode: {self.mode}')
        logger.debug(f'Using mode: {self.mode}')

        self._zero_bound = False
        self._k = self.prior_bounds[self.parameters[0]][1]

    def _rescale_angle(self, x, x_prime, log_j, compute_radius=False,
                       **kwargs):
        angle, lj = rescale_zero_to_one(
            x[self.parameters[0]], *self.prior_bounds[self.parameters[0]])
        log_j += lj
        if self.mode == 'duplicate' or compute_radius:
            angle = np.concatenate([angle, -angle])
            x = np.concatenate([x, x])
            x_prime = np.concatenate([x_prime, x_prime])
            log_j = np.concatenate([log_j, log_j])
        elif self.mode == 'split':
            neg = np.random.choice(angle.size, angle.size // 2, replace=False)
            angle[neg] *= -1

        angle *= self.scale
        return angle, x, x_prime, log_j

    def _inverse_rescale_angle(self, x, x_prime, log_j):
        x[self.parameters[0]], lj = inverse_rescale_zero_to_one(
            np.abs(x[self.parameters[0]]),
            *self.prior_bounds[self.parameters[0]])
        log_j += lj
        return x, x_prime, log_j


class AnglePair(Reparameterisation):
    """Reparameterisation for a pair of angles and a radial component"""
    def __init__(self, parameters=None, prior_bounds=None,
                 prior=None, convention=None):

        self._conventions = ['az-zen', 'ra-dec']

        if len(parameters) not in [2, 3]:
            raise RuntimeError(
                'Must use a pair of angles or a pair plus a radius')

        super().__init__(parameters=parameters, prior_bounds=prior_bounds)

        # Determine which parameter is for the horizontal plane
        # and which is for the vertical plane
        logger.debug('Checking order of parameters')
        b = np.ptp([prior_bounds[p] for p in parameters], axis=1)
        hz = np.where(b == (2 * np.pi))[0][0]
        vt = np.where(b == np.pi)[0][0]

        if len(parameters) == 3:
            r = list({0, 1, 2} - {hz, vt})[0]
            parameters[0], parameters[1], parameters[2] = \
                parameters[hz], parameters[vt], parameters[r]
        else:
            parameters[0], parameters[1] = parameters[hz], parameters[vt]

        if len(parameters) == 2:
            m = '_'.join(parameters)
            parameters.append(f'{m}_radial')
            self.chi = stats.chi(3)
            self.has_prior = True
        else:
            m = '_'.join(parameters)
            self.chi = False

        logger.debug(f'Parameters are: {parameters}')

        self.parameters = parameters
        self.prime_parameters = [f'{m}_{x}' for x in ['x', 'y', 'z']]

        if prior == 'isotropic' and self.chi:
            self.has_prime_prior = True
            logger.info(f'Prime prior enabled for {self.name}')
        else:
            self.has_prime_prior = False
            logger.info(f'Prime prior disabled for {self.name}')

        if convention is None:
            if (self.prior_bounds[self.parameters[1]][0] == 0 and
                    self.prior_bounds[self.parameters[1]][1] == np.pi):
                self.convention = 'az-zen'
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
        return self.parameters[:2]

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
            r * np.sin(x[self.parameters[1]]) * np.sin(x[self.parameters[0]])
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
        x[self.parameters[2]] = np.sqrt(
            x_prime[self.prime_parameters[0]] ** 2. +
            x_prime[self.prime_parameters[1]] ** 2. +
            x_prime[self.prime_parameters[2]] ** 2.)
        x[self.parameters[0]] = \
            np.arctan2(x_prime[self.prime_parameters[1]],
                       x_prime[self.prime_parameters[0]]) % (2. * np.pi)
        x[self.parameters[1]] = \
            np.arctan2(np.sqrt(x_prime[self.prime_parameters[0]] ** 2.
                               + x_prime[self.prime_parameters[1]] ** 2.),
                       x_prime[self.prime_parameters[2]])
        log_j += (-2 * np.log(x[self.parameters[2]])
                  - np.log(np.sin(x[self.parameters[1]])))

        return x, x_prime, log_j

    def _inv_ra_dec(self, x, x_prime, log_j):
        x[self.parameters[2]] = np.sqrt(
            x_prime[self.prime_parameters[0]] ** 2. +
            x_prime[self.prime_parameters[1]] ** 2. +
            x_prime[self.prime_parameters[2]] ** 2.)
        x[self.parameters[0]] = \
            np.arctan2(x_prime[self.prime_parameters[1]],
                       x_prime[self.prime_parameters[0]]) % (2. * np.pi)
        x[self.parameters[1]] = \
            np.arctan2(x_prime[self.prime_parameters[2]],
                       np.sqrt(x_prime[self.prime_parameters[0]] ** 2.
                               + x_prime[self.prime_parameters[1]] ** 2.))

        log_j += (-2 * np.log(x[self.parameters[2]])
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

    def log_prior(self, x):
        """Prior for radial parameter"""
        return self.chi.logpdf(x[self.parameters[2]])

    def x_prime_log_prior(self, x_prime):
        """
        Log probability of 3d Cartesian coordinates for an isotropic
        distribution of angles and a radial component drawn from a chi
        distribution with three degrees of freedom.
        """
        if self.has_prime_prior:
            return - 1.5 * np.log(2 * np.pi) \
                   - np.sum([x_prime[pp] ** 2 for pp in self.prime_parameters],
                            axis=0) / 2
        else:
            raise RuntimeError('x prime prior')


default_reparameterisations = {
    'default': (RescaleToBounds, None),
    'rescaletobounds': (RescaleToBounds, None),
    'rescale-to-bounds': (RescaleToBounds, None),
    'offset': (RescaleToBounds, {'offset': True}),
    'inversion': (RescaleToBounds, {'detect_edges': True,
                                    'boundary_inversion': True,
                                    'inversion_type': 'split'}),
    'inversion-duplicate': (RescaleToBounds, {'detect_edges': True,
                                              'boundary_inversion': True,
                                              'inversion_type': 'duplicate'}),
    'scale': (Rescale, None),
    'rescale': (Rescale, None),
    'angle': (Angle, {}),
    'angle-pi': (Angle, {'scale': 2.0, 'prior': 'uniform'}),
    'angle-2pi': (Angle, {'scale': 1.0, 'prior': 'uniform'}),
    'angle-sine': (Angle, {'scale': 1.0, 'prior': 'sine'}),
    'angle-pair': (AnglePair, None),
    'to-cartesian': (ToCartesian, None),
    'none': (NullReparameterisation, None),
    'null': (NullReparameterisation, None),
    None: (NullReparameterisation, None),
}
