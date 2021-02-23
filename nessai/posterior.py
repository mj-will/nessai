# -*- coding: utf-8 -*-
"""
Functions realted to computing the evidence and posterior samples.
"""
import numpy as np


def logsubexp(x, y):
    """
    Helper function to compute the exponential
    of a difference between two numbers

    Computes: ``x + np.log1p(-np.exp(y-x))``

    Parameters
    ----------
    x, y : float or array_like
        Inputs
    """
    if np.any(x < y):
        raise RuntimeError('cannot take log of negative number '
                           f'{str(x)!s} - {str(y)!s}')

    return x + np.log1p(-np.exp(y - x))


def log_integrate_log_trap(log_func, log_support):
    """
    Trapezoidal integration of given log(func). Returns log of the integral.

    Parameters
    ----------
    log_func : array_like
        Log values of the function to integrate over.
    log_support : array_like
        Log-evidences for each value.

    Returns
    -------
    float
        Log of the result of the integral.
    """
    log_func_sum = np.logaddexp(log_func[:-1], log_func[1:]) - np.log(2)
    log_dxs = logsubexp(log_support[:-1], log_support[1:])

    return np.logaddexp.reduce(log_func_sum + log_dxs)


def compute_weights(samples, nlive):
    """
    Returns the log-evidence and log-weights for the log-likelihood samples
    assumed to the result of nested sampling with nlive live points

    Parameters
    ----------
    samples : array_like
        Log-likelihood samples.
    nlive : int
        Number of live points used in nested sampling.

    Returns
    -------
    float
        The computed log-evidence.
    array_like
        Array of computed weigths (already normalised by the log-evidence).
    """
    samples = np.asarray(samples)
    start_data = np.concatenate(([float('-inf')], samples[:-nlive]))
    end_data = samples[-nlive:]

    log_wts = np.zeros(samples.shape[0])

    log_vols_start = np.cumsum(np.ones(len(start_data) + 1)
                               * np.log1p(-1. / nlive)) - np.log1p(-1 / nlive)
    log_vols_end = np.zeros(len(end_data))
    log_vols_end[-1] = np.NINF
    log_vols_end[0] = log_vols_start[-1] + np.log1p(-1.0 / nlive)
    for i in range(len(end_data) - 1):
        log_vols_end[i+1] = log_vols_end[i] + np.log1p(-1.0 / (nlive - i))

    log_likes = np.concatenate((start_data, end_data, [end_data[-1]]))

    log_vols = np.concatenate((log_vols_start, log_vols_end))
    log_ev = log_integrate_log_trap(log_likes, log_vols)

    log_dXs = logsubexp(log_vols[:-1], log_vols[1:])
    log_wts = log_likes[1:-1] + log_dXs[:-1]

    log_wts -= log_ev

    return log_ev, log_wts


def draw_posterior_samples(nested_samples, nlive):
    """
    Draw posterior samples given the nested samples and number of live points.

    Parameters
    ----------
    nested_samples : structured array
        Array of nested samples.
    nlive : int
        Number of live points used during nested sampling.

    Returns
    -------
    array_like
        Samples from the posterior distribution.
    """
    log_Z, log_w = compute_weights(nested_samples['logL'], nlive)
    log_w -= np.max(log_w)
    log_u = np.log(np.random.rand(nested_samples.size))
    return nested_samples[log_w > log_u]
