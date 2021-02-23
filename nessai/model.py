# -*- coding: utf-8 -*-
"""
Object for defining the use-defined model.
"""
import numpy as np

from .livepoint import (
        parameters_to_live_point,
        numpy_array_to_live_points,
        get_dtype,
        DEFAULT_FLOAT_DTYPE
        )


class Model:
    """Base class for the user-defined model being sampled.

    The user must define the attributes ``names`` ``bounds`` and the metods
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

    names = []
    bounds = {}
    reparameterisations = None
    likelihood_evaluations = 0
    _lower = None
    _upper = None

    @property
    def dims(self):
        """Number of dimensions in the model"""
        if self.names:
            return len(self.names)
        else:
            return None

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
        Computes the proposal probabaility for a new point.

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

    def log_prior(self, x):
        """
        Returns log-prior, must be defined by the user.
        """
        pass

    def log_likelihood(self, x):
        """
        Returns the log-likelihood, must be defined by the user.
        """
        pass

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
        if not self.names:
            raise ValueError('Names for model parameters are not set')
        if not self.bounds:
            raise ValueError('Bounds are not set for model')

        for n in self.names:
            if n not in self.bounds.keys():
                raise RuntimeError(f'Missing bounds for {n}')

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
                raise RuntimeError('Could not draw valid point from within '
                                   'the prior after 10000 tries, check the '
                                   'log prior function.')

        if self.log_prior(x) is None:
            raise RuntimeError('Log-prior function did not return'
                               'a prior value')
        if self.log_likelihood(x) is None:
            raise RuntimeError('Log-likehood function did not return'
                               'a likelihood value')
