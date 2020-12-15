
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


def log_2d_cartesian_prior_sine(x, y):
    """
    Log probability of Cartesian coordinates for a sine distibution of
    angles and a radial component drawn from a chi distribution
    with two degrees of freedom.
    """
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


def log_spin_prior(s1x, s1y, s1z, s2x, s2y, s2z, k1=0.99, k2=0.99):
    """
    Log probability of the prior on the components of spin vectors
    assume the distribution of a_i is uniform.
    """
    r1 = s1x ** 2 + s1y ** 2 + s1z ** 2
    r2 = s2x ** 2 + s2y ** 2 + s2z ** 2
    log_p = np.log(r1 <= (k1 ** 2)) + np.log(r2 <= (k2 ** 2))
    log_p += (-np.log(16 * np.pi ** 2 * k1 * k2) - np.log(r1) - np.log(r2))
    return log_p


def log_spin_prior_uniform(s1x, s1y, s1z, s2x, s2y, s2z, k1=0.99, k2=0.99):
    """
    Log probability of the prior on the components of spin vectors
    assume the distribution the components in uniform within the 2-ball.
    """
    r1 = s1x ** 2 + s1y ** 2 + s1z ** 2
    r2 = s2x ** 2 + s2y ** 2 + s2z ** 2
    log_p = np.log(r1 <= (k1 ** 2)) + np.log(r2 <= (k2 ** 2))
    log_p -= np.log(16 * np.pi ** 2 * k1 * k2 / 9)
    return log_p
