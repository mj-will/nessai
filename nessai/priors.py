"""
Definitions of common priors in the prime space.
"""
import numpy as np


def log_uniform_prior(x, xmin=-1, xmax=1):
    """Unformalised log probability of uniform prior"""
    return np.log((x >= xmin) & (x <= xmax))


def log_2d_cartesian_prior(x, y, k=np.pi):
    """
    Log probability of Cartesian coordinates for a uniform distibution of
    angles on [0, k] and a radial component drawn from a chi distribution
    with two degrees of freedom.
    """
    return - np.log(k) - ((x ** 2 + y ** 2) / 2)


def log_2d_cartesian_prior_sine(x, y, k=np.pi):
    """
    Log probability of Cartesian coordinates for a sine distibution of
    angles and a radial component drawn from a chi distribution
    with two degrees of freedom.
    """
    if k != np.pi:
        raise RuntimeError('x prime prior is incompatible with k != pi')
    r = x ** 2 + y ** 2
    y[y < 0] = 0
    return np.log(y / 2) - 0.5 * np.log(r) - (r / 2)


def log_3d_cartesian_prior(x, y, z):
    """
    Log probability of 3d Cartesian coordinates for an isotropic distribution
    of angles and a radial component drawn from a chi distribution with three
    degrees of freedom.
    """
    return - 1.5 * np.log(2 * np.pi) - (x ** 2 + y ** 2 + z ** 2) / 2
