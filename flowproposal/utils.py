import numpy as np
from scipy.stats import chi

def random_surface_nsphere(ndims, r=1, N=1000):
    """
    Draw N points uniformly on an n-sphere of radius r

    See Marsaglia (1972)
    """
    x = np.random.randn(N, ndims)
    R = np.sqrt(np.sum(x ** 2., axis=0))
    z = x / R
    return r * z.T

def draw_random_nsphere(ndims r=1, N=1000, fuzz=1.0):
    """
    Draw N points uniformly within an n-sphere of radius r
    """
    x = random_surface_nsphere(ndims, r=1, N=N)
    R = np.random.uniform(0, 1, N)
    z = R ** (1 / self.ndims) * x.T
    return fuzz * r * z.T

def draw_truncated_gaussian(ndims r2, N=1000, fuzz=1.0):
    """
    Draw N points from a truncated gaussian with a given radius sqaured
    """
    r = np.sqrt(r2) * fuzz
    p = np.empty([0])
    while p.shape[0] < N:
        p = np.concatenate([p, chi.rvs(ndims, size=N)])
        p = p[p < r]
    x = np.random.randn(p.size, ndims)
    points = (p * x.T / np.sqrt(np.sum(x**2., axis=1))).T
    return points
