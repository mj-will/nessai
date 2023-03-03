# -*- coding: utf-8 -*-
"""
Utilities related to statistics.
"""
import numpy as np
from scipy.special import logsumexp, betainc


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


def weighted_quantile(
    values, quantiles, log_weights=None, values_sorted=False
):
    """Compute quantiles for an array of values.

    Uses the Harrell-Davis quantile estimator.

    Parameters
    ----------
    values : array_like
        Array of values
    quantiles : float or array_like
        Quantiles to compute
    log_weights : array_like, optional
        Array of log-weights
    values_sorted : bool
        If the values are pre-sorted or not

    Returns
    -------
    np.ndarray
        Array of values for each quantile.

    Raises
    ------
    ValueError
        If the effective sample size is not finite.
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    if log_weights is None:
        log_weights = np.zeros(len(values))
    log_weights = np.asarray(log_weights)

    if not (np.all(quantiles >= 0) and np.all(quantiles <= 1)):
        raise ValueError("Quantiles should be in [0, 1]")

    if not values_sorted:
        idx = np.argsort(values)
        values = values[idx]
        log_weights = log_weights[idx]

    log_weights = log_weights - logsumexp(log_weights)
    neff = np.exp(-logsumexp(2 * log_weights))
    weights = np.exp(log_weights)
    if not np.isfinite(neff):
        raise ValueError(
            "Effective sample size is not finite, cannot compute weighted "
            "quantile."
        )
    a = quantiles * (neff + 1)
    b = (1 - quantiles) * (neff + 1)

    end_points = np.zeros((len(weights) + 1, 1), dtype=weights.dtype)
    np.cumsum(weights, out=end_points[1:, 0], axis=0)
    end_points /= end_points[-1]

    # Regularised incomplete beta function
    b_li = betainc(a, b, end_points[:-1])
    b_ri = betainc(a, b, end_points[1:])

    w_star = b_ri - b_li
    if values.ndim == 1:
        values = np.expand_dims(values, -1)
    q_p = np.sum(w_star * values, axis=0)

    return q_p
