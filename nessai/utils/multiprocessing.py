# -*- coding: utf-8 -*-
"""
Utilities related to multiprocessing.
"""
import logging
import multiprocessing

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

    Raise an error if the start method is not `fork`.
    """
    start_method = multiprocessing.get_start_method()
    if start_method != "fork":
        raise RuntimeError(
            "nessai only supports multiprocessing using the 'fork' start "
            f"method. Actual start method is: {start_method}. See the "
            "multiprocessing documentation for more details."
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
