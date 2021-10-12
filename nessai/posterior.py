# -*- coding: utf-8 -*-
"""
Functions related to computing the posterior samples.
"""
import numpy as np

from .evidence import logsubexp, log_integrate_log_trap


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
        Array of computed weights (already normalised by the log-evidence).
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
