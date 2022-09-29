# -*- coding: utf-8 -*-
"""
Functions for computing posterior weights.
"""
import numpy as np
from ..evidence import log_integrate_log_trap, logsubexp


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
    start_data = np.concatenate(([float("-inf")], samples[:-nlive]))
    end_data = samples[-nlive:]

    log_wts = np.zeros(samples.shape[0])

    log_vols_start = np.cumsum(
        np.ones(len(start_data) + 1) * np.log1p(-1.0 / nlive)
    ) - np.log1p(-1 / nlive)
    log_vols_end = np.zeros(len(end_data))
    log_vols_end[-1] = np.NINF
    log_vols_end[0] = log_vols_start[-1] + np.log1p(-1.0 / nlive)
    for i in range(len(end_data) - 1):
        log_vols_end[i + 1] = log_vols_end[i] + np.log1p(-1.0 / (nlive - i))

    log_likes = np.concatenate((start_data, end_data, [end_data[-1]]))

    log_vols = np.concatenate((log_vols_start, log_vols_end))
    log_ev = log_integrate_log_trap(log_likes, log_vols)

    log_dXs = logsubexp(log_vols[:-1], log_vols[1:])
    log_wts = log_likes[1:-1] + log_dXs[:-1]

    log_wts -= log_ev

    return log_ev, log_wts
