import logging
import numpy as np
from scipy.stats import chi

logger = logging.getLogger(__name__)


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


def replace_in_list(target_list, targets, replacements):
    """
    Replace (in place) an entry in a list with a given element
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
        raise ValueError('Target(s) not in target list')

    for t, r in zip(targets, replacements):
        i = target_list.index(t)
        target_list[i] = r
