# -*- coding: utf-8 -*-
"""
Definitions of common priors in the prime space.
"""
import numpy as np


def log_uniform_prior(x, xmin=-1, xmax=1):
    """Unformalised log probability of uniform prior.

    Parameters
    ----------
    x : array_like
        Parameter to computed log-prior for
    xmin : float, optional
        Lower bound on prior
    xmax : float, optional
        Upper bound on prior
    """
    return np.log((x >= xmin) & (x <= xmax))


def log_2d_cartesian_prior(x, y, k=np.pi):
    """
    Log probability for isotropic 2d Cartesian coordinates.

    Assumes a uniform distribution of angles on [0, k] and a radial component
    drawn from a chi distribution with two degrees of freedom.

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates
    k : float
        Range over which the angles used to obtain the Cartesian coordinates
        are defined.
    """
    return -np.log(k) - ((x**2 + y**2) / 2)


def log_2d_cartesian_prior_sine(x, y, k=np.pi):
    """
    Log probability of Cartesian coordinates for a angle with a sine prior

    Assumes angles drawn for a sine distribution andand a radial component
    drawn from a chi distribution with two degrees of freedom.

    Raises a RuntimeError if the anlges were not defined on the range [0, pi].

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates
    k : float
        Must be ``np.pi``. Included for compatibility with the interface for
        angle reparameterisations.
    """
    if k != np.pi:
        raise RuntimeError("x prime prior is incompatible with k != pi")
    r = x**2 + y**2
    y[y < 0] = 0
    return np.log(y / 2) - 0.5 * np.log(r) - (r / 2)


def log_3d_cartesian_prior(x, y, z):
    """
    Log probability for 3d isotropic Cartesian coordinates.

    Assumes an isotropic distribution of angles and a radial component drawn
    from a chi distribution with three degrees of freedom.

    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates
    """
    return -1.5 * np.log(2 * np.pi) - (x**2 + y**2 + z**2) / 2
