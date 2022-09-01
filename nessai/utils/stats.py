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
