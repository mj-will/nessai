# -*- coding: utf-8 -*-
"""
Utilities related to statistics.
"""
import numpy as np
from scipy.special import logsumexp


def effective_sample_size(log_w):
    """Compute Kish's effective sample size.

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


def effective_volume(
    log_w: np.ndarray, log_l: np.ndarray, beta: float
) -> float:
    """Compute the effective volume.

    Use for annealing in the importance nested sampler.

    Parameters
    ----------
    log_w
        Log weights (ratio of prior and meta proposal).
    log_l
        Log-likelihood values.
    beta
        Annealing value.

    Returns
    -------
    float
        The effective volume.
    """
    lr = np.exp(beta * (log_l - log_l.max()))
    w = np.exp(log_w)
    return np.sum(w * lr) / np.sum(w)


def rolling_mean(x, N=10):
    """Compute the rolling mean with a given window size.

    Based on this answer from StackOverflow: \
        https://stackoverflow.com/a/47490020

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array of samples
    N : int
        Size of the window over which the rolling mean is computed.

    Returns
    -------
    :obj:`numpy.ndarray`
        Array containing the moving average.
    """
    # np.cumsum is faster but doesn't work with infs in the data.
    return np.convolve(
        np.pad(x, (N // 2, N - 1 - N // 2), mode="edge"),
        np.ones(N) / N,
        mode="valid",
    )


def weighted_quantile(values, quantiles, weights=None, values_sorted=False):
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
        raise ValueError("Quantiles should be in [0, 1]")

    if not values_sorted:
        idx = np.argsort(values)
        values = values[idx]
        weights = weights[idx]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)
