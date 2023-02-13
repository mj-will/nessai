# -*- coding: utf-8 -*-
"""
Utilities related to drawing samples.
"""
import numpy as np
from scipy import stats
from scipy.special import gammaincinv


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
    ndarray
        Array of samples with shape (N, dims)
    """
    x = np.random.randn(N, dims)
    R = np.sqrt(np.sum(x**2.0, axis=1))[:, np.newaxis]
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
    ndarray
        Array of samples with shape (N, dims)
    """
    x = draw_surface_nsphere(dims, r=1, N=N)
    R = np.random.uniform(0, 1, (N, 1))
    z = R ** (1 / dims) * x
    return fuzz * r * z


def draw_uniform(dims, r=(1,), N=1000, fuzz=1.0):
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
    ndarray
        Array of samples with shape (N, dims)
    """
    return np.random.randn(N, dims)


def draw_truncated_gaussian(dims, r, N=1000, fuzz=1.0, var=1):
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
    sigma = np.sqrt(var)
    u_max = stats.chi.cdf(r * fuzz / sigma, df=dims)
    u = np.random.uniform(0, u_max, N)
    p = sigma * stats.chi.ppf(u, df=dims)
    x = np.random.randn(p.size, dims)
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
    ) -> None:
        self.dims = dims
        self.radius = radius
        self.fuzz = fuzz
        self.chi = stats.chi(df=self.dims)

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
        u = self.u_max * np.random.rand(N)
        # Inverse CDF of a chi-distribution
        p = np.sqrt(2 * gammaincinv(0.5 * self.dims, u))
        x = np.random.randn(self.dims, N)
        points = (p * x / np.sqrt(np.sum(x**2.0, axis=0))).T
        return points
