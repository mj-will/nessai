# -*- coding: utf-8 -*-
"""
Object for defining the use-defined model.
"""
from abc import ABC, abstractmethod
import datetime
import logging
import numpy as np

from . import config
from .livepoint import (
    empty_structured_array,
    parameters_to_live_point,
    numpy_array_to_live_points,
    unstructured_view,
    _unstructured_view_dtype,
)
from .utils.multiprocessing import (
    get_n_pool,
    log_likelihood_wrapper,
    log_prior_wrapper,
    batch_evaluate_function,
    check_vectorised_function,
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
    """

    _names = None
    _bounds = None
    _dtype = None
    reparameterisations = None
    """
    dict
        Dictionary of reparameterisations that overrides the values specified.
    """
    likelihood_evaluations = 0
    """
    int
        Number of likelihood evaluations.
    """
    likelihood_evaluation_time = datetime.timedelta()
    """
    :py:obj:`datetime.timedelta()`
        Time spent evaluating the likelihood.
    """
    _lower = None
    _upper = None
    pool = None
    """
    obj
        Multiprocessing pool for evaluating the log-likelihood.
    """
    allow_vectorised_prior = True
    """
    bool
        Allow the model to use a vectorised prior. If True, nessai will
        try to check if the log-prior is vectorised and use call the method
        as a vectorised function. If False, nessai won't check and, even if the
        log-prior is vectorised, it will only evaluate the log-prior one
        sample at a time.
    """
    allow_vectorised = True
    """
    bool
        Allow the model to use a vectorised likelihood. If True, nessai will
        try to check if the model is vectorised and use call the likelihood
        as a vectorised function. If False, nessai won't check and, even if the
        likelihood is vectorised, it will only evaluate the likelihood one
        sample at a time.
    """
    likelihood_chunksize = None
    """
    int
        Chunksize to use with a vectorised likelihood. If specified the
        likelihood will be called with at most chunksize points at once.
    """
    allow_multi_valued_likelihood = False
    """
    bool
        Allow for a multi-valued likelihood function that will return different
        likelihood values for the same point in parameter space. This is only
        recommended when the variation is significantly smaller that the
        variations in the likelihood across the prior.
    """
    parallelise_prior = False
    """
    bool
        Parallelise calculating the log-prior using the multiprocessing pool.
    """
    _vectorised_likelihood = None
    _vectorised_prior = None
    _pool_configured = False
    n_pool = None

    @property
    def names(self):
        """List of the names of each parameter in the model."""
        if self._names is None:
            raise RuntimeError("`names` is not set!")
        return self._names

    @names.setter
    def names(self, names):
        if not isinstance(names, list):
            raise TypeError("`names` must be a list")
        elif not names:
            raise ValueError("`names` list is empty!")
        elif len(names) == 1:
            raise OneDimensionalModelError(
                "names list has length 1. "
                "nessai is not designed to handle one-dimensional models due "
                "to limitations imposed by the normalising flow-based "
                "proposals it uses. Consider using other methods instead of "
                "nessai."
            )
        else:
            self._names = names

    @property
    def bounds(self):
        """Dictionary with the lower and upper bounds for each parameter."""
        if self._bounds is None:
            raise RuntimeError("`bounds` is not set!")
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if not isinstance(bounds, dict):
            raise TypeError("`bounds` must be a dictionary.")
        elif len(bounds) == 1:
            raise OneDimensionalModelError(
                "bounds dictionary has length 1. "
                "nessai is not designed to handle one-dimensional models due "
                "to limitations imposed by the normalising flow-based "
                "proposals it uses. Consider using other methods instead of "
                "nessai."
            )
        elif not all([len(b) == 2 for b in bounds.values()]):
            raise ValueError("Each entry in `bounds` must have length 2.")
        else:
            self._bounds = {p: np.asarray(b) for p, b in bounds.items()}

    @property
    def dims(self):
        """Number of dimensions in the model"""
        d = len(self.names)
        if d == 0:
            d = None
        return d

    def _set_upper_lower(self):
        """Set the upper and lower bounds arrays"""
        bounds_array = np.array([self.bounds[n] for n in self.names])
        self._lower = bounds_array[:, 0]
        self._upper = bounds_array[:, 1]

    @property
    def lower_bounds(self):
        """Lower bounds on the priors"""
        if self._lower is None:
            self._set_upper_lower()
        return self._lower

    @property
    def upper_bounds(self):
        """Upper bounds on the priors"""
        if self._upper is None:
            self._set_upper_lower()
        return self._upper

    @property
    def vectorised_likelihood(self):
        """Boolean to indicate if the likelihood is vectorised or not.

        Checks that the values returned by computing the likelihood for
        individual samples matches those return by evaluating the likelihood
        in a batch. If a TypeError or ValueError are raised the likelihood is
        assumed to be vectorised.

        This check can be prevented by setting
        :py:attr:`nessai.model.Model.allowed_vectorised` to ``False``.
        """
        if self._vectorised_likelihood is None:
            if self.allow_vectorised:
                # Avoids calling prior on multiple points
                x = np.concatenate([self.new_point() for _ in range(10)])
                self._vectorised_likelihood = check_vectorised_function(
                    self.log_likelihood,
                    x,
                    dtype=config.livepoints.logl_dtype,
                )
            else:
                self._vectorised_likelihood = False
        return self._vectorised_likelihood

    @vectorised_likelihood.setter
    def vectorised_likelihood(self, value):
        """Manually set the value for vectorised likelihood."""
        self._vectorised_likelihood = value

    @property
    def vectorised_prior(self):
        if self._vectorised_prior is None:
            if self.allow_vectorised_prior:
                # Avoids calling prior on multiple points
                x = np.concatenate([self.new_point() for _ in range(10)])
                self._vectorised_prior = check_vectorised_function(
                    self.log_prior,
                    x,
                    dtype=config.livepoints.default_float_dtype,
                )
            else:
                self._vectorised_prior = False
        return self._vectorised_prior

    @vectorised_prior.setter
    def vectorised_prior(self, value):
        """Manually set the value for vectorised prior."""
        self._vectorised_prior = value

    @property
    def _view_dtype(self):
        """dtype used for unstructured view"""
        if self._dtype is None:
            x = empty_structured_array(0, self.names)
            self._dtype = _unstructured_view_dtype(x, self.names)
        return self._dtype

    def configure_pool(self, pool=None, n_pool=None):
        """Configure a multiprocessing pool for the likelihood computation.

        Configuration will be skipped if the pool has already been configured.

        Parameters
        ----------
        pool :
            User provided pool. Must call
            :py:func:`nessai.utils.multiprocessing.initialise_pool_variables`
            before creating the pool or pass said function to the initialiser
            with the model.
        n_pool : int
            Number of threads to use to create an instance of
            :py:obj:`multiprocessing.Pool`.
        """
        if self._pool_configured:
            logger.warning("Multiprocessing pool has already been configured.")
            return
        self.pool = pool
        self.n_pool = n_pool
        if self.pool:
            logger.info("Using user specified pool")
            n_pool = get_n_pool(self.pool)
            if n_pool is None and not self.n_pool:
                logger.warning(
                    "Could not determine number of processes in pool and "
                    "user has not specified the number. Likelihood "
                    "vectorisation will be disabled."
                )
                self.allow_vectorised = False
            elif n_pool:
                self.n_pool = n_pool
                logger.debug(f"User pool has {n_pool} processes")
        elif self.n_pool:
            logger.info(
                f"Starting multiprocessing pool with {n_pool} processes"
            )
            import multiprocessing
            from nessai.utils.multiprocessing import (
                check_multiprocessing_start_method,
                initialise_pool_variables,
            )

            check_multiprocessing_start_method()

            self.pool = multiprocessing.Pool(
                processes=self.n_pool,
                initializer=initialise_pool_variables,
                initargs=(self,),
            )
        else:
            logger.info("pool and n_pool are none, no multiprocessing pool")
        self._pool_configured = True

    def close_pool(self, code=None):
        """Close the the multiprocessing pool.

        Also resets the pool configuration.
        """
        if getattr(self, "pool", None) is not None:
            logger.info("Starting to close worker pool.")
            if code == 2:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Finished closing worker pool.")
        self._pool_configured = False

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
        while logP == -np.inf:
            p = parameters_to_live_point(
                np.random.uniform(
                    self.lower_bounds, self.upper_bounds, self.dims
                ),
                self.names,
            )
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
        new_points = empty_structured_array(N, names=self.names)
        n = 0
        while n < N:
            p = numpy_array_to_live_points(
                np.random.uniform(
                    self.lower_bounds, self.upper_bounds, [N, self.dims]
                ),
                self.names,
            )
            p = p[np.isfinite(self.log_prior(p))]
            m = min(p.size, N - n)
            new_points[n : (n + m)] = p[:m]
            n += m
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
        return ~np.any(
            [
                (x[n] < self.bounds[n][0]) | (x[n] > self.bounds[n][1])
                for n in self.names
            ],
            axis=0,
        )

    def in_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Check if samples are within the unit hypercube

        Parameters
        ----------
        x : numpy.ndarray
            Structured array of live points. Must contain all of the parameters
            in the model.

        Returns
        -------
        Array of bool
            Array with the same length as x where True indicates the point
            is within the prior bounds.
        """
        x = self.unstructured_view(x)
        return ~np.any((x > 1) | (x < 0), axis=-1)

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
        raise NotImplementedError("User must implement this method!")

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

    def sample_unit_hypercube(self, n: int = 1) -> np.ndarray:
        """ "Sample from the unit hypercube.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        numpy.ndarray
            Structured array of samples
        """
        return numpy_array_to_live_points(
            np.random.rand(n, self.dims),
            names=self.names,
        )

    def from_unit_hypercube(self, x):
        """Map from the unit hypercube to the priors.

        Not implemented by default.
        """
        raise NotImplementedError

    def to_unit_hypercube(self, x):
        """Map from the prior space to the unit hypercube.

        Not implemented by default.
        """
        raise NotImplementedError

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
        self.likelihood_evaluations += x.size
        return self.log_likelihood(x)

    def batch_evaluate_log_likelihood(
        self, x: np.ndarray, unit_hypercube: bool = False
    ) -> np.ndarray:
        """Evaluate the likelihood for a batch of samples.

        Uses the pool if available.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Array of samples
        unit_hypercube : bool
            Indicates if input samples are from the unit hypercube or not.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of log-likelihood values
        """
        st = datetime.datetime.now()
        if unit_hypercube:
            x = self.from_unit_hypercube(x)
        log_likelihood = batch_evaluate_function(
            self.log_likelihood,
            x,
            self.allow_vectorised and self.vectorised_likelihood,
            chunksize=self.likelihood_chunksize,
            func_wrapper=log_likelihood_wrapper,
            pool=self.pool,
            n_pool=self.n_pool,
        )
        self.likelihood_evaluations += x.size
        self.likelihood_evaluation_time += datetime.datetime.now() - st
        return log_likelihood.astype(config.livepoints.logl_dtype)

    def batch_evaluate_log_prior(
        self, x: np.ndarray, unit_hypercube: bool = False
    ) -> np.ndarray:
        """Evaluate the log-prior for a batch of samples.

        Uses the pool if available.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Array of samples
        unit_hypercube : bool
            Indicates if input samples are from the unit hypercube or not.


        Returns
        -------
        :obj:`numpy.ndarray`
            Array of log-prior values
        """
        if unit_hypercube:
            x = self.from_unit_hypercube(x)
        return batch_evaluate_function(
            self.log_prior,
            x,
            self.allow_vectorised_prior and self.vectorised_prior,
            func_wrapper=log_prior_wrapper,
            pool=self.pool if self.parallelise_prior else None,
            n_pool=self.n_pool,
        )

    def unstructured_view(self, x):
        """An unstructured view of point(s) x that only contains the \
            parameters in the model.

        This is quicker than converting to a unstructured array and does not
        create a copy of the array.

        Calls :py:func:`nessai.livepoint.unstructured_view` with a pre-computed
        dtype.

        .. warning::

            Will only work if all of the model parameters use the same dtype.

        Parameters
        ----------
        x : structured_array
            Structured array of points

        Returns
        -------
        numpy.ndarray
            View of x as an unstructured array that contains only the
            parameters in the model. Shape is (x.size, self.dims).
        """
        return unstructured_view(x, dtype=self._view_dtype)

    def verify_model(self):
        """
        Verify that the model is correctly setup. This includes checking
        the names, bounds and log-likelihood.

        Returns
        -------
        bool
            True if the model was verified as valid.
        """
        if not isinstance(self.names, list):
            raise TypeError("`names` must be a list")

        if not isinstance(self.bounds, dict):
            raise TypeError("`bounds` must be a dictionary")

        if not self.names:
            raise ValueError(
                f"`names` is not set to a valid value: {self.names}"
            )
        if not self.bounds or not isinstance(self.bounds, dict):
            raise ValueError(
                f"`bounds` is not set to a valid value: {self.bounds}"
            )

        if self.dims == 1:
            raise OneDimensionalModelError(
                "model is one-dimensional. "
                "nessai is not designed to handle one-dimensional models due "
                "to limitations imposed by the normalising flow-based "
                "proposals it uses. Consider using other methods instead of "
                "nessai."
            )

        for n in self.names:
            if n not in self.bounds.keys():
                raise RuntimeError(f"Missing bounds for {n}")

        if (
            np.isfinite(self.lower_bounds).all()
            and np.isfinite(self.upper_bounds).all()
        ):
            logP = -np.inf
            counter = 0
            while (logP == -np.inf) or (logP == np.inf):
                x = numpy_array_to_live_points(
                    np.random.uniform(
                        self.lower_bounds, self.upper_bounds, [1, self.dims]
                    ),
                    self.names,
                )
                logP = self.log_prior(x)
                counter += 1
                if counter == 1000:
                    raise RuntimeError(
                        "Could not draw valid point from within the prior "
                        "after 10000 tries, check the log prior function."
                    )
        else:
            logger.warning("Model has infinite bounds(s)")
            logger.warning("Testing with `new_point`")
            try:
                x = self.new_point(1)
                logP = self.log_prior(x)
            except Exception as e:
                raise RuntimeError(
                    "Could not draw a new point and compute the log prior "
                    f"with error: {e}. \n Check the prior bounds."
                )

        if self.log_prior(x) is None:
            raise RuntimeError(
                "Log-prior function did not return " "a prior value"
            )
        if self.log_likelihood(x) is None:
            raise RuntimeError(
                "Log-likelihood function did not return " "a likelihood value"
            )

        if self.allow_multi_valued_likelihood:
            logger.warning(
                "Multi-valued likelihood is allowed. "
                "This may lead to slow sampling and strange results."
            )
        else:
            logl = np.array([self.log_likelihood(x) for _ in range(16)])
            if not all(logl == logl[0]):
                raise RuntimeError(
                    "Repeated calls to the log-likelihood with the same "
                    "parameters return different values."
                )

        if self.log_prior(x).dtype == np.dtype("float16"):
            logger.warning(
                "log_prior returned an array with float16 precision. "
                "This not recommended and can lead to numerical errors."
                " Consider casting to a higher precision."
            )
        return True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["pool"] = None
        return state
