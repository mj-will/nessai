# -*- coding: utf-8 -*-
"""
Utilities related to multiprocessing.
"""

_model = None


def initialise_pool_variables(model):
    """Prepare the model for use with a multiprocessing pool.

    Makes a global copy of the model. Should be called before initialising
    a pool.

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
