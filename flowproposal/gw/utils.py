import numpy as np
import scipy.stats as stats

def angle_to_cartesian(alpha, r=None, scale=1.0):
    """
    Decompose an angle into a real and imaginary part
    """
    rescaled_alpha = alpha * scale
    if r is None:
        r = stats.chi.rvs(2, size=alpha.size)
    x = r * np.cos(rescaled_alpha)
    y = r * np.sin(rescaled_alpha)
    return x, y, np.log(r)

def cartesian_to_angle(x, y, scale=1.0):
    """
    Reconstruct an angle given the real and imaginary part
    """
    radius = np.sqrt(x ** 2. + y** 2.)
    angle = (np.arctan2(y, x) + 2. * np.pi) % (2. * np.pi) / scale
    return angle, radius, -np.log(radius)

def sky_to_cartesian(ra, dec, dL=None):
    """
    Decompose an angle into a real and imaginary part
    """
    if dL is None:
        dL = stats.chi.rvs(3, size=ra.size)
    # amplitudes
    x = dL * np.cos(dec) * np.cos(ra)
    y = dL * np.cos(dec) * np.sin(ra)
    z = dL * np.sin(dec)
    return x, y, z, 2. * np.log(dL) + np.log(np.cos(dec))

def cartesian_to_sky(x, y, z):
    """
    Reconstruct an angle given the real and imaginary part
    """
    dL = np.sqrt(np.sum([x **2., y**2., z**2.], axis=0))
    dec = np.arctan2(z, np.sqrt(x ** 2. + y ** 2.0))
    ra = np.arctan2(y, x) % (2. * np.pi)
    return ra, dec, dL, - 2. * np.log(dL) - np.log(np.cos(dec))
