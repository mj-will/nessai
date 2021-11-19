# -*- coding: utf-8 -*-
"""Utitilies related to statistics."""
import numpy as np
from scipy.special import logsumexp


def effective_sample_size(log_w) -> float:
    """Compute the Kish effective sample size

    Parameters
    ----------
    log_w : array_like
        Log-weights.

    Returns
    -------
    float
        The effective sample size.
    """
    log_w = np.array(log_w)
    log_w -= logsumexp(log_w)
    n = np.exp(-logsumexp(2 * log_w))
    return n


def weighted_quantile(
    values,
    quantiles,
    weights=None,
    values_sorted=False
):
    """Compute quantiles for an array of values.

    Based on: https://stackoverflow.com/a/29677616

    Parameters
    ----------
    values : array_like
        Array of values
    quantiles : float or array_like
        Quantiles to compute
    weights : array_like, optional
        Array of weights

    Returns
    -------
    np.ndarray
        Array of values for each quantile.
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights)
    if not np.all(quantiles >= 0) and np.all(quantiles <= 1):
        raise ValueError('Quantiles should be in [0, 1]')

    if not values_sorted:
        idx = np.argsort(values)
        values = values[idx]
        weights = weights[idx]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)
