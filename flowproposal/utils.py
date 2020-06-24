import numpy as np
from scipy.stats import chi

def random_surface_nsphere(dims, r=1, N=1000):
    """
    Draw N points uniformly on an n-sphere of radius r

    See Marsaglia (1972)
    """
    x = np.random.randn(N, dims)
    R = np.sqrt(np.sum(x ** 2., axis=0))
    z = x / R
    return r * z.T

def draw_random_nsphere(dims, r=1, N=1000, fuzz=1.0):
    """
    Draw N points uniformly within an n-sphere of radius r
    """
    x = random_surface_nsphere(dims, r=1, N=N)
    R = np.random.uniform(0, 1, N)
    z = R ** (1 / self.dims) * x.T
    return fuzz * r * z.T

def draw_truncated_gaussian(dims, r, N=1000, fuzz=1.0):
    """
    Draw N points from a truncated gaussian with a given radius sqaured
    """
    r *= fuzz
    p = np.empty([0])
    while p.shape[0] < N:
        p = np.concatenate([p, chi.rvs(dims, size=N)])
        p = p[p < r]
    x = np.random.randn(p.size, dims)
    points = (p * x.T / np.sqrt(np.sum(x**2., axis=1))).T
    return points
