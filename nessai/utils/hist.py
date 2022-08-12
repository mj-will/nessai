# -*- coding: utf-8 -*-
"""
Utilities related to histogramming.
"""
import numpy as np


def _hist_bin_fd(x):
    """
    The Freedman-Diaconis histogram bin estimator.

    See original Numpy implementation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_sturges(x):
    """
    Sturges histogram bin estimator.

    See original Numpy implementation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return np.ptp(x) / (np.log2(x.size) + 1.0)


def auto_bins(x, max_bins=50):
    """
    Compute the number bins for a histogram using numpy.histogram_bin_edges
    but enforces a maximum number of bins.

    Parameters
    ----------
    array : array_like
        Input data
    bins : int or sequence of scalars or str, optional
        Method for determining number of bins, see numpy documentation
    max_bins : int, optional (1000)
        Maximum number of bins

    Returns
    -------
    int
        Number of bins
    """
    x = np.asarray(x)
    if not x.size:
        raise RuntimeError("Input array is empty!")
    fd_bw = _hist_bin_fd(x)
    sturges_bw = _hist_bin_sturges(x)
    if fd_bw:
        bw = min(fd_bw, sturges_bw)
    else:
        bw = sturges_bw

    if bw:
        n_bins = int(np.ceil(np.ptp(x)) / bw)
    else:
        n_bins = 1

    nbins = min(n_bins, max_bins)
    return nbins
