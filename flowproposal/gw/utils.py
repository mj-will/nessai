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
    Reconstruct an angle given the real and imaginary part. Assume the angle
    is defined on [0, 2 pi] / scale.
    """
    radius = np.sqrt(x ** 2. + y ** 2.)
    angle = (np.arctan2(y, x) + 2. * np.pi) % (2. * np.pi) / scale
    return angle, radius, -np.log(radius)


def sky_to_cartesian(ra, dec, dL=None):
    """
    Convert right ascension, declination and (optinal) luminosity distance
    (defined on [0, 1]) to Cartesian coordinates.

    Parameters
    ----------
    ra, dec: array_like
        Right ascension and declination
    dL: array_like, optional
        Corresponding luminosity distance defined on [0, 1]. If None (default)
        radial componment is drawn froma chi distribution with 3 degrees of
        freedom

    Returns
    -------
    x, y, z: array_like
        Cartesian coordinates
    log_J: array_like
        Determinant of the log-Jacobian
    """
    if dL is None:
        dL = stats.chi.rvs(3, size=ra.size)
    x = dL * np.cos(dec) * np.cos(ra)
    y = dL * np.cos(dec) * np.sin(ra)
    z = dL * np.sin(dec)
    return x, y, z, 2. * np.log(dL) + np.log(np.cos(dec))


def cartesian_to_sky(x, y, z):
    """
    Reconstruct an angle given the real and imaginary part

    Parameters
    ----------
    x, y, z: array_like
        Three dimensional Cartesian coordinates

    Returns:
    ra, dec: array_like
        Right ascension and declination
    dl: array_like
        Luminosity distance
    log_J: array_like
        Determinant of the log-Jacobian
    """
    dL = np.sqrt(np.sum([x ** 2., y ** 2., z ** 2.], axis=0))
    dec = np.arctan2(z, np.sqrt(x ** 2. + y ** 2.0))
    ra = np.arctan2(y, x) % (2. * np.pi)
    return ra, dec, dL, - 2. * np.log(dL) - np.log(np.cos(dec))


def zero_one_to_cartesian(theta):
    """
    Convert a variable defined on [0,1] to an angle on [-pi, pi] and
    to Cartesian coordinates with a radius drawn from a chi distribution
    with two degrees of freedom. The lower bound is place at 0 and the upper
    bound at -pi/pi.

    Parameters
    ----------
    theta: array_like
        Array of values bound on [0, 1]

    Returns
    -------
    x, y: array_like
        Cartesian coordinates
    log_J: array_like
        Determinant of the log-Jacobian
    """
    neg = np.random.choice(theta.size, theta.size // 2, replace=False)
    theta[neg] *= -1
    return angle_to_cartesian(theta, scale=np.pi)


def cartesian_to_zero_one(x, y):
    """
    Convert Cartesian coordinates to a variable defined on [0, 1] and
    a corresponding radius.

    Parameters
    ----------
    x, y: array_like
        Cartesian coordinates

    Returns
    -------
    theta: array_like
        Variable defined on [0,1]
    radius: array_like
        Corresponding radius
    log_J: array_like
        Determinant of log-Jacobian
    """
    theta = np.abs(np.arctan2(y, x)) / np.pi
    radius = np.sqrt(x ** 2. + y ** 2.)
    return theta, radius, -np.log(radius)
