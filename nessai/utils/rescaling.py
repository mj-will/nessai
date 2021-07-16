# -*- coding: utf-8 -*-
"""
Utilities related to rescaling.
"""
import logging

import numpy as np

from .hist import auto_bins

logger = logging.getLogger(__name__)


def rescale_zero_to_one(x, xmin, xmax):
    """
    Rescale a value to 0 to 1

    Parameters
    ----------
    x : ndarray
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    ndarray
        Array of rescaled values
    ndarray
        Array of log determinants of Jacobians for each sample
    """
    return (x - xmin) / (xmax - xmin), -np.log(xmax - xmin)


def inverse_rescale_zero_to_one(x, xmin, xmax):
    """
    Rescale from 0 to 1 to xmin to xmax

    Parameters
    ----------
    x : ndarray
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    ndarray
        Array of rescaled values
    ndarray
        Array of log determinants of Jacobians for each sample
    """
    return (xmax - xmin) * x + xmin, np.log(xmax - xmin)


def rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale a value to -1 to 1

    Parameters
    ----------
    x : ndarray
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    ndarray
        Array of rescaled values
    ndarray
        Array of log determinants of Jacobians for each sample
    """
    return ((2. * (x - xmin) / (xmax - xmin)) - 1,
            np.log(2) - np.log(xmax - xmin))


def inverse_rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale from -1 to 1 to xmin to xmax

    Parameters
    ----------
    x : ndarray
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    ndarray
        Array of rescaled values
    ndarray
        Array of log determinants of Jacobians for each sample
    """
    return ((xmax - xmin) * ((x + 1) / 2.) + xmin,
            np.log(xmax - xmin) - np.log(2))


def detect_edge(x, x_range=None, percent=0.1, cutoff=0.5, nbins='auto',
                allow_both=False, allow_none=False,
                allowed_bounds=['lower', 'upper'], test=None):
    """
    Detect edges in input distributions based on the density.

    Parameters
    ----------
    x: array_like
        Samples
    x_range : array_like, optional
        Lower and upper bounds used to check inversion, if not specified
        min and max of data are used.
    percent: float (0.1)
        Percentage of interval used to check edges
    cutoff: float (0.1)
        Minimum fraction of the maximum density contained within the
        percentage of the interval specified
    nbins : float or 'auto'
        Number of bins used for histogram.
    allow_both: bool
        Allow function to return both instead of force either upper or lower
    allow_none: bool
        Allow for neither lower or upper bound to be returned
    allowed_bounds : list
        List of alloweds bounds.
    test : str or None
        If not None this skips the process and just returns the value of test.
        This is used to verify the inversion in all possible scenarios.

    Returns
    -------
    str or False, {'lower', 'upper', 'both', False}
        Returns the boundary to apply the inversion or False is no inversion
        is to be applied
    """
    bounds = ['lower', 'upper']
    if test is not None:
        logger.debug('Using test in detect_edge')
        if test in bounds and test not in allowed_bounds:
            logger.debug(f'{test} is not an allowed bound, returning False')
            return False
        else:
            return test
    if not all(b in bounds for b in allowed_bounds):
        raise RuntimeError(f'Unknown allowed bounds: {allowed_bounds}')
    if nbins == 'auto':
        nbins = auto_bins(x)

    hist, bins = np.histogram(x, bins=nbins, density=True, range=x_range)
    n = max(int(len(bins) * percent), 1)
    bounds_fraction = \
        np.array([np.sum(hist[:n]), np.sum(hist[-n:])]) * (bins[1] - bins[0])
    max_idx = np.argmax(hist)
    max_density = hist[max_idx] * (bins[1] - bins[0])
    logger.debug(f'Max. density: {max_density:.3f}')

    for i, b in enumerate(bounds):
        if b not in allowed_bounds:
            bounds.pop(i)
            bounds_fraction = np.delete(bounds_fraction, i)
    if max_idx <= n and 'lower' in bounds:
        return bounds[0]
    elif max_idx >= (len(bins) - n) and 'upper' in bounds:
        return bounds[-1]
    elif not np.any(bounds_fraction > cutoff * max_density) and allow_none:
        logger.debug('Density too low at both bounds')
        return False
    else:
        if (np.all(bounds_fraction > cutoff * max_density) and allow_both and
                len(bounds) > 1):
            logger.debug('Both bounds above cutoff')
            return 'both'
        else:
            return bounds[np.argmax(bounds_fraction)]


def configure_edge_detection(d, detect_edges):
    """
    Configure parameters for edge detection

    Parameters
    ----------
    d : dict
        Dictionary of kwargs passed to detect_edge.
    detect_edges : bool
        If true allows for no inversion to be applied.

    Returns
    -------
    dict
        Updated kwargs
    """
    default = dict(cutoff=0.5)
    if d is None:
        d = {}
    if detect_edges:
        d['allow_none'] = True
    else:
        d['allow_none'] = False
        d['cutoff'] = 0.0
    default.update(d)
    logger.debug(f'detect edges kwargs: {default}')
    return default


def determine_rescaled_bounds(prior_min, prior_max, x_min, x_max, invert,
                              offset=0, rescale_bounds=[-1, 1],
                              inversion=False):
    """
    Determine the values of the prior min and max in the rescaled
    space.

    Parameters
    ----------
    prior_min : float
        Mininum of the prior.
    prior_max : float
        Maximum of the prior.
    x_min : float
        New minimum.
    x_max : float
        New maximum.
    invert : False or {'upper', 'lower', 'both'}
        Type of inversion.
    """
    if x_min == x_max:
        raise RuntimeError('New minimum and maximum are equal')
    scale = rescale_bounds[1] - rescale_bounds[0]
    shift = rescale_bounds[0]
    lower = scale * (prior_min - offset - x_min) / (x_max - x_min) + shift
    upper = scale * (prior_max - offset - x_min) / (x_max - x_min) + shift
    if not inversion:
        return lower, upper
    elif (not invert or invert is None):
        return 2 * lower - 1, 2 * upper - 1
    elif invert == 'upper':
        return lower - 1, 1 - lower
    elif invert == 'lower':
        return -upper, upper
    elif invert == 'both':
        return -0.5, 1.5
    else:
        raise RuntimeError


def logit(x, fuzz=1e-2):
    """
    Logit function that also returns log Jacobian

    Parameters
    ----------
    x : ndarray
        Array of values
    fuzz : float, optional
        Fuzz used to avoid nans in logit. Values are rescaled from [0, 1]
        to [0-fuzz, 1+fuzz]. By default no fuzz is applied
    """
    x += fuzz
    x /= (1 + 2 * fuzz)
    return np.log(x) - np.log(1 - x), -np.log(np.abs(x - x ** 2))


def sigmoid(x, fuzz=1e-2):
    """
    Sigmoid function that also returns log Jacobian

    Parameters
    ----------
    x : ndarray
        Array of values
    fuzz : float, optional
        Fuzz used to avoid nans in logit
    """
    x = np.asarray(x)
    x = np.divide(1, 1 + np.exp(-x))
    log_J = np.log(np.abs(x - x ** 2))
    x *= (1 + 2 * fuzz)
    x -= fuzz
    return x, log_J


rescaling_functions = {'logit': (logit, sigmoid)}