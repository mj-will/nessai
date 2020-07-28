import json
import logging
import numpy as np
from scipy.stats import chi
import torch

from nflows.distributions.uniform import BoxUniform

logger = logging.getLogger(__name__)


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
    #if not dims == len(r):
    #    raise RuntimeError('Dimensions and bounds for hypercube do not match')
    #r *= fuzz
    # Any of the bounds are greater than one, set them to one
    #r = np.min([r, np.ones(dims)], axis=0)
    #return np.random.uniform(1-r, r, (N, dims))
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
        p = np.concatenate([p, chi.rvs(dims, size=N)])
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
    return (x - xmin ) / (xmax - xmin), -np.log(xmax - xmin)


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
    return (2. * (x - xmin ) / (xmax - xmin)) - 1, np.log(2) - np.log(xmax - xmin)


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
    return (xmax - xmin) * ((x + 1) / 2.) + xmin, np.log(xmax - xmin) - np.log(2)


def setup_logger(output=None, label=None, log_level='INFO'):
    """
    Setup logger

    Based on the implementation in Bilby: https://git.ligo.org/michael.williams/bilby/-/blob/master/bilby/core/utils.py

    Parameters
    ----------
    output : str, optional
        Path of to output directory
    label : str, optional
        Label for this instance of the logger
    log_level: {'ERROR', 'WARNING', 'INFO', 'DEBUG'}
        Level of logging parsed to logger
    """
    import os
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

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%m-%d %H:%M'))
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
    """Class to encode numpy arrays when saving as json"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def counter(fn):
        def wrapper(*args, **kwargs):
            wrapper.calls += 1
            return fn(*args, **kwargs)
        wrapper.calls= 0
        wrapper.__name__= fn.__name__
        return wrapper
