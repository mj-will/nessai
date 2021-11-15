# -*- coding: utf-8 -*-
"""
Object for defining the use-defined model.
"""
from abc import ABC, abstractmethod
import logging
import numpy as np

from .livepoint import (
        parameters_to_live_point,
        numpy_array_to_live_points,
        get_dtype,
        DEFAULT_FLOAT_DTYPE
        )


logger = logging.getLogger(__name__)


class OneDimensionalModelError(Exception):
    """Exception raised when the model is one-dimensional"""
    pass


class Model(ABC):
    """Base class for the user-defined model being sampled.

    The user must define the attributes ``names`` ``bounds`` and the methods
    ``log_likelihood`` and ``log_prior``.

    The user can also define the reparemeterisations here instead of in
    the keyword arguments passed to the sampler.


    Attributes
    ----------
    names : list of str
        List of names of parameters, e.g. ['x', 'y']
    bounds : dict
        Dictionary of prior bounds, e.g. {'x': [-1, 1], 'y': [-1, 1]}
    reparameterisations : dict
        Dictionary of reparameterisations that overrides the values specified
        with keyword arguments.
    likelihood_evaluations : int
        Number of likelihood evaluations
    """

    _names = None
    _bounds = None
    reparameterisations = None
    likelihood_evaluations = 0
    _lower = None
    _upper = None

    @property
    def names(self):
        """List of the names of each parameter in the model."""
        if self._names is None:
            raise RuntimeError('`names` is not set!')
        return self._names

    @names.setter
    def names(self, names):
        if not isinstance(names, list):
            raise TypeError('`names` must be a list')
        elif not names:
            raise ValueError('`names` list is empty!')
        elif len(names) == 1:
            raise OneDimensionalModelError(
                'names list has length 1. '
                'nessai is not designed to handle one-dimensional models due '
                'to limitations imposed by the normalising flow-based '
                'proposals it uses. Consider using other methods instead of '
                'nessai.'
            )
        else:
            self._names = names

    @property
    def bounds(self):
        """Dictionary with the lower and upper bounds for each parameter."""
        if self._bounds is None:
            raise RuntimeError('`bounds` is not set!')
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if not isinstance(bounds, dict):
            raise TypeError('`bounds` must be a dictionary.')
        elif len(bounds) == 1:
            raise OneDimensionalModelError(
                'bounds dictionary has length 1. '
                'nessai is not designed to handle one-dimensional models due '
                'to limitations imposed by the normalising flow-based '
                'proposals it uses. Consider using other methods instead of '
                'nessai.'
            )
        elif not all([len(b) == 2 for b in bounds.values()]):
            raise ValueError('Each entry in `bounds` must have length 2.')
        else:
            self._bounds = {p: np.asarray(b) for p, b in bounds.items()}

    @property
    def dims(self):
        """Number of dimensions in the model"""
        d = len(self.names)
        if d == 0:
            d = None
        return d

    @property
    def lower_bounds(self):
        """Lower bounds on the priors"""
        if self._lower is None:
            bounds_array = np.array(list(self.bounds.values()))
            self._lower = bounds_array[:, 0]
            self._upper = bounds_array[:, 1]
        return self._lower

    @property
    def upper_bounds(self):
        """Upper bounds on the priors"""
        if self._upper is None:
            bounds_array = np.array(list(self.bounds.values()))
            self._lower = bounds_array[:, 0]
            self._upper = bounds_array[:, 1]
        return self._upper

    def new_point(self, N=1):
        """
        Create a new LivePoint, drawn from within bounds.

        See `new_point_log_prob` if changing this method.

        Parameters
        ----------
        N : int, optional
            Number of points to draw. By default draws one point. If N > 1
            points are drawn using a faster method.

        Returns
        -------
        ndarray
            Numpy structured array with fields for each parameter
            and log-prior (logP) and log-likelihood (logL)
        """
        if N > 1:
            return self._multiple_new_points(N)
        else:
            return self._single_new_point()

    def new_point_log_prob(self, x):
        """
        Computes the proposal probability for a new point.

        This does not assume the that points will be drawn according to the
        prior. If `new_point` is redefined this method must be updated to
        match.

        Parameters
        ----------
        x : ndarray
            Points in a structured array

        Returns
        -------
        ndarray
            Log proposal probability for each point
        """
        return np.zeros(x.size)

    def _single_new_point(self):
        """
        Draw a single point within the prior

        Returns
        -------
        ndarray
            Numpy structured array with fields for each parameter
            and log-prior (logP) and log-likelihood (logL)
        """
        logP = -np.inf
        while (logP == -np.inf):
            p = parameters_to_live_point(
                    np.random.uniform(self.lower_bounds, self.upper_bounds,
                                      self.dims),
                    self.names)
            logP = self.log_prior(p)
        return p

    def _multiple_new_points(self, N):
        """
        Draw multiple points within the prior. Draws N points and accepts
        only those for which log-prior is finite.

        Parameters
        ----------
        N : int
            Number of points to draw

        Returns
        -------
        np.ndarray
            Numpy structured array with fields for each parameter
            and log-prior (logP) and log-likelihood (logL)
        """
        new_points = np.array([], dtype=get_dtype(self.names,
                                                  DEFAULT_FLOAT_DTYPE))
        while new_points.size < N:
            p = numpy_array_to_live_points(
                    np.random.uniform(self.lower_bounds, self.upper_bounds,
                                      [N, self.dims]),
                    self.names)
            logP = self.log_prior(p)
            new_points = np.concatenate([new_points, p[np.isfinite(logP)]])
        return new_points

    def in_bounds(self, x):
        """Check if a set of live point are within the prior bounds.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Structured array of live points. Must contain all of the parameters
            in the model.

        Returns
        -------
        Array of bool
            Array with the same length as x where True indicates the point
            is within the prior bounds.
        """
        return ~np.any([(x[n] < self.bounds[n][0]) | (x[n] > self.bounds[n][1])
                        for n in self.names], axis=0)

    def sample_parameter(self, name, n=1):
        """Draw samples for a specific parameter from the prior.

        Should be implemented by the user and return a numpy array of length
        n. The array should NOT be a structured array. This method is not
        required for standard sampling with nessai. It is intended for use
        with :py:class:`nessai.conditional.ConditionalFlowProposal`.

        Parameters
        ----------
        name : str
            Name for the parameter to sample
        n : int, optional
            Number of samples to draw.
        """
        raise NotImplementedError('User must implement this method!')

    def parameter_in_bounds(self, x, name):
        """
        Check if an array of values for specific parameter are in the prior \
            bounds.

        Parameters
        ----------
        x : :obj:`numpy:ndarray`
            Array of values. Not a structured array.

        Returns
        -------
        Array of bool
            Array with the same length as x where True indicates the value
            is within the prior bounds.
        """
        return (x >= self.bounds[name][0]) & (x <= self.bounds[name][1])

    @abstractmethod
    def log_prior(self, x):
        """
        Returns log-prior, must be defined by the user.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, x):
        """
        Returns the log-likelihood, must be defined by the user.
        """
        raise NotImplementedError

    def evaluate_log_likelihood(self, x):
        """
        Evaluate the log-likelihood and track the number of calls.

        Returns
        -------
        float
            Log-likelihood value

        """
        self.likelihood_evaluations += 1
        return self.log_likelihood(x)

    def verify_model(self):
        """
        Verify that the model is correctly setup. This includes checking
        the names, bounds and log-likelihood.
        """
        if not isinstance(self.names, list):
            raise TypeError('`names` must be a list')

        if not isinstance(self.bounds, dict):
            raise TypeError('`bounds` must be a dictionary')

        if not self.names:
            raise ValueError(
                f'`names` is not set to a valid value: {self.names}'
            )
        if not self.bounds or not isinstance(self.bounds, dict):
            raise ValueError(
                f'`bounds` is not set to a valid value: {self.bounds}'
            )

        if self.dims == 1:
            raise OneDimensionalModelError(
                'model is one-dimensional. '
                'nessai is not designed to handle one-dimensional models due '
                'to limitations imposed by the normalising flow-based '
                'proposals it uses. Consider using other methods instead of '
                'nessai.'
            )

        for n in self.names:
            if n not in self.bounds.keys():
                raise RuntimeError(f'Missing bounds for {n}')

        if (np.isfinite(self.lower_bounds).all() and
                np.isfinite(self.upper_bounds).all()):
            logP = -np.inf
            counter = 0
            while (logP == -np.inf) or (logP == np.inf):
                x = numpy_array_to_live_points(
                        np.random.uniform(self.lower_bounds, self.upper_bounds,
                                          [1, self.dims]),
                        self.names)
                logP = self.log_prior(x)
                counter += 1
                if counter == 1000:
                    raise RuntimeError(
                        'Could not draw valid point from within the prior '
                        'after 10000 tries, check the log prior function.')
        else:
            logger.warning('Model has infinite bounds(s)')
            logger.warning('Testing with `new_point`')
            try:
                x = self.new_point(1)
                logP = self.log_prior(x)
            except Exception as e:
                raise RuntimeError(
                    'Could not draw a new point and compute the log prior '
                    f'with error: {e}. \n Check the prior bounds.')

        if self.log_prior(x) is None:
            raise RuntimeError('Log-prior function did not return '
                               'a prior value')
        if self.log_likelihood(x) is None:
            raise RuntimeError('Log-likelihood function did not return '
                               'a likelihood value')
