# -*- coding: utf-8 -*-
"""
Utilities related to multiprocessing.
"""
import logging
import multiprocessing

import numpy as np
from nessai.utils.structures import array_split_chunksize
from nessai.config import livepoints

_model = None
logger = logging.getLogger(__name__)


def get_n_pool(pool):
    """Determine the number of processes in a multiprocessing pool.

    Parameters
    ----------
    pool : object
        Multiprocessing pool or similar.

    Returns
    -------
    int or None
        Number of processes. Returns None if number could not be determined.
    """
    try:
        n_pool = pool._processes
    except AttributeError:
        try:
            n_pool = len(pool._actor_pool)
        except AttributeError:
            n_pool = None
            logger.warning(
                "Could not determine number of processes in pool of type: "
                f"{type(pool)}."
            )
    return n_pool


def check_multiprocessing_start_method():
    """Check the multiprocessing start method.

    Print a warning if the start method is not `fork`.
    """
    start_method = multiprocessing.get_start_method()
    if start_method != "fork":
        logger.warning(
            f"Using {start_method} start method for multiprocessing. "
            "This may lead to high memory usage or errors. "
            "Consider using the `fork` start method. "
            "See the multiprocessing documentation for more details."
        )


def initialise_pool_variables(model):
    """Prepare the model for use with a multiprocessing pool.

    Makes a global copy of the model. Should be called before initialising
    a pool or passed to the :code:`initializer` argument with the model as one
    of the :code:`initargs`.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        Model to be copied to a global variable.
    """
    global _model
    _model = model


def log_likelihood_wrapper(x):
    """Wrapper for the log-likelihood for use with multiprocessing.

    Should be used alongside
    :py:func:`nessai.utils.multiprocessing.initialise_pool_variables`

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array of samples.

    Returns
    -------
    :obj:`numpy.ndarray`
        Array of log-likelihoods.
    """
    return _model.log_likelihood(x)


def log_prior_wrapper(x):
    """Wrapper for the log-prior for use with multiprocessing.

    Should be used alongside
    :py:func:`nessai.utils.multiprocessing.initialise_pool_variables`

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array of samples.

    Returns
    -------
    :obj:`numpy.ndarray`
        Array of log-prior values.
    """
    return _model.log_prior(x)


def batch_evaluate_function(
    func,
    x,
    vectorised,
    chunksize=None,
    pool=None,
    n_pool=None,
    func_wrapper=None,
):
    """Evaluate a function over a batch of inputs.

    Parameters
    ----------
    func : Callable
        The function to evaluate.
    x : numpy.ndarray
        The values over which to evaluate the function.
    vectorised : bool
        Boolean to indicate if the function is vectorised or not.
    chunksize : Optional[int]
        Maximum number of inputs that will be passed to the function at once.
        Only applies if the function is vectorised.
    pool : Optional[multiprocessing.Pool]
        Multiprocessing pool used to evaluate the function.
    n_pool : Optional[int]
        Number of cores in the multiprocessing. Determines how values are split
        when calling :code:`Pool.map`.
    func_wrapper : Optional[Callable]
        Wrapper to the function to use instead of the function when using the
        pool.

    Returns
    -------
    numpy.ndarray
        Array of outputs
    """
    if pool is None:
        if vectorised:
            if chunksize:
                out = np.concatenate(
                    list(map(func, array_split_chunksize(x, chunksize)))
                )
            else:
                out = func(x)
        else:
            out = np.array(
                [func(xx) for xx in x],
            ).flatten()
    else:
        if func_wrapper is None:
            func_wrapper = func
        if vectorised:
            if chunksize:
                out = np.concatenate(
                    pool.map(func_wrapper, array_split_chunksize(x, chunksize))
                )
            else:
                out = np.concatenate(
                    pool.map(func_wrapper, np.array_split(x, n_pool))
                )
        else:
            out = np.array(pool.map(func_wrapper, x)).flatten()
    return out


def check_vectorised_function(func, x, dtype=None, atol=1e-15, rtol=1e-15):
    """Check if a function is vectorised given a set of inputs.

    Parameters
    ----------
    func : Callable
        Function to test
    x : numpy.ndarray
        Inputs over which to evaluate the function
    dtype :  Optional[Union[str, numpy.dtype]]
        Dtype to cast the results to. If not specified, the default dtype
        for livepoints is used.
    rtol : float
        Relative tolerance, see numpy documentation for :code:`allclose`.
    atol : float
        Abosoulte tolerance, see numpy documentation for :code:`allclose`.

    Returns
    -------
    numpy.ndarray
        Array of outputs
    """
    if dtype is None:
        dtype = livepoints.default_float_dtype
    if len(x) <= 1:
        raise ValueError("Input has length <= 1")

    target = np.array([func(xx) for xx in x], dtype=dtype)

    try:
        batch = func(x).astype(dtype)
    except (TypeError, ValueError, AttributeError):
        logger.debug("Assuming function is not vectorised")
        return False
    else:
        if np.allclose(target, batch, atol=atol, rtol=rtol):
            logger.debug("Function is vectorised")
            return True
        else:
            logger.debug("Individual and batch likelihoods are not equal")
            logger.debug(f"Individual: {target}")
            logger.debug(f"Batch: {batch}")
            return False
