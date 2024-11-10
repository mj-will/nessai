# -*- coding: utf-8 -*-
"""
Utilities related to drawing samples.
"""

import logging

import numpy as np
from scipy import stats
from scipy.special import gammaincinv

logger = logging.getLogger(__name__)


def compute_radius(n, q=0.95):
    """Compute the radius that contains a fraction of the total probability \
         in an n-dimensional unit Gaussian.

    Uses the inverse CDF of a chi-distribution with n degrees of freedom.

    Parameters
    ----------
    n : int
        Number of dimensions
    q : float
        Fraction of the total probability

    Returns
    -------
    float
        Radius
    """
    return stats.chi.ppf(q, n)


def draw_surface_nsphere(dims, r=1, N=1000, rng=None):
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
    ndarray
        Array of samples with shape (N, dims)
    """
    if rng is None:
        logger.debug("No rng specified, using the default rng.")
        rng = np.random.default_rng()
    x = rng.standard_normal((N, dims))
    R = np.sqrt(np.sum(x**2.0, axis=1))[:, np.newaxis]
    z = x / R
    return r * z


def draw_nsphere(dims, r=1, N=1000, fuzz=1.0, rng=None):
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
    ndarray
        Array of samples with shape (N, dims)
    """
    if rng is None:
        logger.debug("No rng specified, using the default rng.")
        rng = np.random.default_rng()
    x = draw_surface_nsphere(dims, r=1, N=N, rng=rng)
    R = rng.random((N, 1))
    z = R ** (1 / dims) * x
    return fuzz * r * z


def draw_uniform(dims, r=(1,), N=1000, fuzz=1.0, rng=None):
    """
    Draw from a uniform distribution on [0, 1].

    Deals with extra input parameters used by other draw functions

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-ball. (Ignored by this function)
    N : int, ignored
        Number of samples to draw
    fuzz : float, ignored
        Fuzz factor by which to increase the radius of the n-ball. (Ignored by
        this function)

    Returns
    -------
    ndarraay
        Array of samples with shape (N, dims)
    """
    if rng is None:
        logger.debug("No rng specified, using the default rng.")
        rng = np.random.default_rng()
    return rng.random((N, dims))


def draw_gaussian(dims, r=1, N=1000, fuzz=1.0, rng=None):
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
    ndarray
        Array of samples with shape (N, dims)
    """
    if rng is None:
        logger.debug("No rng specified, using the default rng.")
        rng = np.random.default_rng()
    return rng.standard_normal((N, dims))


def draw_truncated_gaussian(dims, r, N=1000, fuzz=1.0, var=1, rng=None):
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
    ndarray
        Array of samples with shape (N, dims)
    """
    if rng is None:
        logger.debug("No rng specified, using the default rng.")
        rng = np.random.default_rng()
    sigma = np.sqrt(var)
    u_max = stats.chi.cdf(r * fuzz / sigma, df=dims)
    u = rng.uniform(0, u_max, N)
    p = sigma * stats.chi.ppf(u, df=dims)
    x = rng.standard_normal((p.size, dims))
    points = (p * x.T / np.sqrt(np.sum(x**2.0, axis=1))).T
    return points


class NDimensionalTruncatedGaussian:
    """Class for sampling from a radially truncated n-dimensional Gaussian

    Parameters
    ----------
    dims :
        The number of dimensions
    radius :
        The radius for the truncation
    fuzz : float
        The fuzz factor
    """

    def __init__(
        self,
        dims: int,
        radius: float,
        fuzz: float = 1.0,
        rng: np.random.Generator = None,
    ) -> None:
        self.dims = dims
        self.radius = radius
        self.fuzz = fuzz
        self.chi = stats.chi(df=self.dims)
        if rng is None:
            logger.debug("No rng specified, using the default rng.")
            rng = np.random.default_rng()
        self.rng = rng
        self.u_max = self.chi.cdf(self.radius * self.fuzz)

    def sample(self, N: int) -> np.ndarray:
        """Sample from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw

        Returns
        -------
        numpy.ndarray
            Array of samples of shape [n, dims].
        """
        u = self.u_max * self.rng.random(N)
        # Inverse CDF of a chi-distribution
        p = np.sqrt(2 * gammaincinv(0.5 * self.dims, u))
        x = self.rng.standard_normal((self.dims, N))
        points = (p * x / np.sqrt(np.sum(x**2.0, axis=0))).T
        return points
