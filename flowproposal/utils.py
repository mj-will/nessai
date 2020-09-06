import json
import logging
import os
import shutil

from nflows.distributions.uniform import BoxUniform
import numpy as np
from scipy import stats, spatial
import torch

logger = logging.getLogger(__name__)


def logit(x):
    """
    Logit function that also returns log Jacobian

    Parameters
    ----------
    x: array_like
    """
    return np.log(x) - np.log(1 - x), -np.log(np.abs(x - x ** 2))


def sigmoid(x):
    """
    Sigmoid function that also returns log Jacobian

    Parameters
    ----------
    x: array_like
    """
    return (1 / (1 + np.exp(-x)),
            np.log(np.abs(np.exp(-x) / (np.exp(-x) + 1) ** 2)))


def compute_indices_ks_test(indices, nlive, mode='D+'):
    """
    Compute the two-sided KS test for discrete insertion indices for a given
    number of live points

    Parameters
    ----------
    indices: array_like
        Indices of newly inserteed live points
    nlive: int
        Number of live points

    Returns
    ------
    D: float
        Two-sided KS statistic
    p: float
        p-value
    """
    if len(indices):
        u, counts = np.unique(indices, return_counts=True)
        analytic_cdf = u / nlive
        cdf = np.cumsum(counts) / len(indices)
        if mode == 'D+':
            D = np.max(cdf - analytic_cdf)
        elif mode == 'D-':
            D = np.max(analytic_cdf[1:] - cdf[:-1])
        else:
            raise RuntimeError
        p = stats.ksone.sf(D, nlive)
        return D, p
    else:
        return None, None


def draw_surface_nsphere(dims, r=1, N=1000):
    """
    Draw N points uniformly from  n-1 sphere of radius r using Marsaglia's
    algorithm. E.g for 3 dimensions returns points on a 'regular' sphere.

    See Marsaglia (1972)

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-sphere, if specified it is used to rescale the samples
    N : int, optional
        Number of samples to draw

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    x = np.random.randn(N, dims)
    R = np.sqrt(np.sum(x ** 2., axis=1))[:, np.newaxis]
    z = x / R
    return r * z


def draw_nsphere(dims, r=1, N=1000, fuzz=1.0):
    """
    Draw N points uniformly within an n-sphere of radius r

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-ball
    N : int, optional
        Number of samples to draw
    fuzz : float, optional
        Fuzz factor by which to increase the radius of the n-ball

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    x = draw_surface_nsphere(dims, r=1, N=N)
    R = np.random.uniform(0, 1, (N, 1))
    z = R ** (1 / dims) * x
    return fuzz * r * z


def get_uniform_distribution(dims, r):
    """
    Return a Pytorch distribution that is uniform in the number of
    dims specified
    """
    r = r * torch.ones(dims)
    return BoxUniform(low=-r, high=r)


def draw_uniform(dims, r=(1,), N=1000, fuzz=1.0):
    """
    Draw from the
    """
    return np.random.uniform(0, 1, (N, dims))


def draw_gaussian(dims, r=1, N=1000, fuzz=1.0):
    """
    Wrapper for numpy.random.randn that deals with extra input parameters
    r and fuzz

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-ball
    N : int, ignored
        Number of samples to draw
    fuzz : float, ignored
        Fuzz factor by which to increase the radius of the n-ball

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    return np.random.randn(N, dims)


def draw_truncated_gaussian(dims, r, N=1000, fuzz=1.0):
    """
    Draw N points from a truncated gaussian with a given a radius

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float
        Radius of the truncated Gaussian
    N : int, ignored
        Number of samples to draw
    fuzz : float, ignored
        Fuzz factor by which to increase the radius of the truncated Gaussian

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    r *= fuzz
    p = np.empty([0])
    while p.shape[0] < N:
        p = np.concatenate([p, stats.chi.rvs(dims, size=N)])
        p = p[p < r]
    x = np.random.randn(p.size, dims)
    points = (p * x.T / np.sqrt(np.sum(x**2., axis=1))).T
    return points


def replace_in_list(target_list, targets, replacements):
    """
    Replace (in place) an entry in a list with a given element

    Parameters
    ----------
    target_list : list
        List to update
    targets : list
        List of items to update
    replacements : list
        List of replacement items
    """
    if not isinstance(targets, list):
        if isinstance(targets, int):
            targets = [targets]
        else:
            targets = list(targets)
    if not isinstance(replacements, list):
        if isinstance(replacements, int):
            replacements = [replacements]
        else:
            replacements = list(replacements)

    if not all([t in target_list for t in targets]):
        raise ValueError(f'Target(s) not in target list: {targets}')

    for t, r in zip(targets, replacements):
        i = target_list.index(t)
        target_list[i] = r


def rescale_zero_to_one(x, xmin, xmax):
    """
    Rescale a value to 0 to 1

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return (x - xmin) / (xmax - xmin), -np.log(xmax - xmin)


def inverse_rescale_zero_to_one(x, xmin, xmax):
    """
    Rescale from 0 to 1 to xmin to xmax

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return (xmax - xmin) * x + xmin, np.log(xmax - xmin)


def rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale a value to -1 to 1

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return ((2. * (x - xmin) / (xmax - xmin)) - 1,
            np.log(2) - np.log(xmax - xmin))


def inverse_rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale from -1 to 1 to xmin to xmax

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return ((xmax - xmin) * ((x + 1) / 2.) + xmin,
            np.log(xmax - xmin) - np.log(2))


def detect_edge(x, bounds, percent=0.1, cutoff=0.1, nbins='fd',
                both=False, allow_none=False):
    """
    Detect edges in input distributions based on the density.

    Checks if data is uniform over the interval specified by the bounds and if
    the data is normally distributed about the mid-point of the bounds

    Parameters
    ----------
    x: array_like
        Samples
    bounds: list
        Lower and upper bound
    percent: float (0.1)
        Percentage of interval used to check edges
    cutoff: float (0.1)
        Minimum fraction of the maximum density contained within the
        percentage of the interval specified
    both: bool
        Allow function to return both instead of force either upper or lower
    allow_none: bool
        Allow for neither lower or upper bound to be returned
    """
    hist, bins = np.histogram(x, bins=nbins, density=True)
    n = int(len(bins) * percent)
    bounds_fraction = \
        np.array([np.sum(hist[:n]), np.sum(hist[-n:])]) * (bins[1] - bins[0])
    uniform_p = stats.kstest(x, 'uniform', args=(bounds[0], np.ptp(bounds)))[1]
    normal_p = stats.kstest(x, 'norm', args=(np.sum(bounds) / 2,))[1]
    max_density = hist.max()
    logger.debug('Max. density: {max_density:.3f}')
    if uniform_p >= 0.05 and both:
        logger.debug('Samples pass KS test for uniform')
        return 'both'
    elif normal_p >= 0.05 and allow_none:
        logger.debug('Samples pass KS test for normal distribution')
        return False
    elif not np.any(bounds_fraction > cutoff * max_density) and allow_none:
        logger.debug('Density too low at both bounds')
        return False
    else:
        if np.all(bounds_fraction > cutoff * max_density) and both:
            logger.debug('Both bounds above cutoff')
            return 'both'
        else:
            bound = np.argmax(bounds_fraction)
            if bound == 0:
                return 'lower'
            elif bound == 1:
                return 'upper'
            else:
                raise RuntimeError('Bounds were not computed correctly')


def compute_minimum_distances(samples, metric='euclidean'):
    """
    Compute the distance to the nearest neighbour of each sample

    Parameters
    ----------
    samples: array_like
        Array of samples
    """
    d = spatial.distance.cdist(samples, samples, metric)
    d[d == 0] = np.nan
    dmin = np.nanmin(d, axis=1)
    return dmin


def setup_logger(output=None, label='label', log_level='INFO'):
    """
    Setup logger

    Based on the implementation in Bilby

    Parameters
    ----------
    output : str, optional
        Path of to output directory
    label : str, optional
        Label for this instance of the logger
    log_level: {'ERROR', 'WARNING', 'INFO', 'DEBUG'}
        Level of logging parsed to logger
    """
    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger('flowproposal')
    logger.propagate = False
    logger.setLevel(level)

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) \
            is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s',
            datefmt='%m-%d %H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
        if label:
            if output:
                if not os.path.exists(output):
                    os.makedirs(output, exist_ok=True)
            else:
                output = '.'
            log_file = '{}/{}.log'.format(output, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger


class NumpyEncoder(json.JSONEncoder):
    """
    Class to encode numpy arrays when saving as json

    Based on: https://stackoverflow.com/a/57915246
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def safe_file_dump(data, filename, module, save_existing=False):
    """ Safely dump data to a .pickle file

    See Bilby for the original impletmentation:
    https://git.ligo.org/michael.williams/bilby/-/blob/master/bilby/core/utils.py

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """
    if save_existing:
        if os.path.exists(filename):
            old_filename = filename + ".old"
            shutil.move(filename, old_filename)

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    shutil.move(temp_filename, filename)
