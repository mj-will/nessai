import logging
import numpy as np
from scipy import stats, interpolate

from ..utils import (
    logit,
    sigmoid,
    rescale_zero_to_one,
    inverse_rescale_zero_to_one
)

logger = logging.getLogger(__name__)

try:
    import lalsimulation as lalsim
    from lal import MSUN_SI as m_sol
except ImportError:
    logger.debug(
        'Could not import LAL functions, running with reduced functionality')

try:
    from astropy import cosmology as cosmo
    import astropy.units as u
except ImportError:
    logger.debug(
        'Could not import astropy, running with reduced functionality')


def angle_to_cartesian(alpha, r=None, scale=1.0):
    """
    Decompose an angle into a real and imaginary part
    """
    alpha = np.asarray(alpha)
    rescaled_alpha = alpha * scale
    if r is None:
        r = stats.chi.rvs(2, size=alpha.size)
    elif any(r < 0):
        raise RuntimeError('Radius cannot be negative.')
    x = r * np.cos(rescaled_alpha)
    y = r * np.sin(rescaled_alpha)
    return x, y, np.log(r)


def cartesian_to_angle(x, y, scale=1.0, zero='centre'):
    """
    Reconstruct an angle given the real and imaginary part. Assume the angle
    is defined on [0, 2 pi] / scale.

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates
    scale : float, optional
        Rescaling factor used to rescale from [0, 2pi]
    zero : str, {centre, bound}
        Specifiy is zero should be the central value or lower bound
    """
    radius = np.sqrt(x ** 2. + y ** 2.)
    if zero == 'centre':
        angle = np.arctan2(y, x) / scale
    else:
        angle = (np.arctan2(y, x) + 2. * np.pi) % (2. * np.pi) / scale
    return angle, radius, -np.log(radius)


def ra_dec_to_cartesian(ra, dec, dL=None):
    """
    Convert right ascension, declination and (optional) luminosity distance
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


def cartesian_to_ra_dec(x, y, z):
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


def azimuth_zenith_to_cartesian(azimuth, zenith, dL=None):
    """
    Convert azimuth, zenith and (optional) luminosity distance
    (defined on [0, 1]) to Cartesian coordinates.

    Parameters
    ----------
    azimuth, zenith: array_like
        Azimuth and zenith
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
        dL = stats.chi.rvs(3, size=azimuth.size)
    x = dL * np.sin(zenith) * np.cos(azimuth)
    y = dL * np.sin(zenith) * np.sin(azimuth)
    z = dL * np.cos(zenith)
    return x, y, z, 2. * np.log(dL) + np.log(np.sin(zenith))


def cartesian_to_azimuth_zenith(x, y, z):
    """
    Reconstruct an angle given the real and imaginary part

    Parameters
    ----------
    x, y, z: array_like
        Three dimensional Cartesian coordinates

    Returns:
    --------
    azimuth, zenith: array_like
        Azimuth and zenith
    dl: array_like
        Luminosity distance
    log_J: array_like
        Determinant of the log-Jacobian
    """
    dL = np.sqrt(np.sum([x ** 2., y ** 2., z ** 2.], axis=0))
    zenith = np.arctan2(np.sqrt(x ** 2. + y ** 2.0), z)
    azimuth = np.arctan2(y, x) % (2. * np.pi)
    return azimuth, zenith, dL, - 2. * np.log(dL) - np.log(np.sin(zenith))


def zero_one_to_cartesian(theta, mode='split'):
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
    theta = theta.copy()
    if mode == 'duplicate':
        theta = np.concatenate([theta, -theta])
    elif mode == 'split':
        neg = np.random.choice(theta.size, theta.size // 2, replace=False)
        theta[neg] *= -1
    elif mode == 'half':
        pass
    else:
        raise RuntimeError(f'Unknown mode: {mode}')
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


def _spin_jacobian_to_precessing(sx, sy, sz):
    """Log Jacobian for transformation from spin vector to angles"""
    return - 0.5 * np.log(sx ** 2 + sy ** 2) \
           - 0.5 * np.log(sx ** 2 + sy ** 2 + sz ** 2)


def _spin_jacobian_from_precessing(a, theta):
    """Log jacobian for tranformation from precessing to spin vector"""
    return 2 * np.log(a) + np.log(np.sin(theta))


@np.vectorize
def transform_from_precessing_parameters(theta_jn, phi_jl, theta_1, theta_2,
                                         phi_12, a_1, a_2, m1, m2, f_ref,
                                         phase):
    f_ref = float(f_ref)
    m1 *= m_sol
    m2 *= m_sol
    iota, s1x, s1y, s1z, s2x, s2y, s2z = \
        lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, theta_1, theta_2, phi_12, a_1, a_2, m1, m2,
                f_ref, phase)

    log_J = _spin_jacobian_from_precessing(a_1, theta_1)
    log_J += _spin_jacobian_from_precessing(a_2, theta_2)
    return iota, s1x, s1y, s1z, s2x, s2y, s2z, log_J


@np.vectorize
def transform_to_precessing_parameters(iota, s1x, s1y, s1z, s2x, s2y, s2z,
                                       m1, m2, f_ref, phase):
    f_ref = float(f_ref)
    theta_jn, phi_jl, theta_1, theta_2, phi_12, a_1, a_2 = \
        lalsim.SimInspiralTransformPrecessingWvf2PE(
            iota, s1x, s1y, s1z, s2x, s2y, s2z, m1, m2, f_ref, phase)

    log_J = _spin_jacobian_to_precessing(s1x, s1y, s1z)
    log_J += _spin_jacobian_to_precessing(s2x, s2y, s2z)
    return theta_jn, phi_jl, theta_1, theta_2, phi_12, a_1, a_2, log_J


def rescale_and_logit(x, xmin, xmax):
    x, log_J = rescale_zero_to_one(x, xmin, xmax)
    x, log_J_logit = logit(x)
    return x, log_J + log_J_logit


def rescale_and_sigmoid(x, xmin, xmax):
    x, log_J_sig = sigmoid(x)
    x, log_J = inverse_rescale_zero_to_one(x, xmin, xmax)
    return x, log_J + log_J_sig


class DistanceConverter:
    """Base object for converting from a distance parameter to a uniform
    parameter.
    """
    has_conversion = False
    has_Jacobian = False

    def to_uniform_parameter(self, d):
        raise NotImplementedError

    def from_uniform_parameter(self, d):
        raise NotImplementedError


class NullDistanceConverter(DistanceConverter):

    has_Jacobian = True

    def __init__(self, d_min=None, d_max=None, **kwargs):
        if kwargs:
            logger.warning(f'Kwargs {kwargs} will be ignored for distance')

    def to_uniform_parameter(self, d):
        return d, np.zeros_like(d)

    def from_uniform_parameter(self, d):
        return d. np.zeros_like(d)


class PowerLawConverter(DistanceConverter):
    """Convert from a distance parameter sample from a power law to a uniform
    parameter

    Assumes d proportional d^power.

    Parameters
    ----------
    power : float
        Power to use for the power-law.
    scale : float
        Factor used to rescale distance prior to convetering to the uniform
        parameter.
    """
    has_conversion = True
    has_jacobian = True

    def __init__(self, power=None, scale=1000.0, **kwargs):
        if power is None:
            raise RuntimeError(
                'Must specify the power to use in the power-law')
        self.power = power
        self.scale = scale
        self._power = self.power + 1

        if self._power == 3:
            self._f = np.cbrt
        elif self._power == 2:
            self._f = np.sqrt
        else:
            self._f = lambda x: x ** (1 / self._power)

    def _log_jacobian(self, d):
        return - self._power * np.log(self.scale) + np.log(self._power) + \
                (self._power - 1) * np.log(d)

    def _log_jacobian_inv(self, d):
        return np.log(self.scale) - np.log(self._power) + \
                (1 / self._power - 1) * np.log(d)

    def to_uniform_parameter(self, d):
        """Convert distance to a parameter with a uniform prior"""
        return (d / self.scale) ** (self._power), self._log_jacobian(d)

    def from_uniform_parameter(self, d):
        """Convert to distance from a parameter that has a uniform prior"""
        return self.scale * self._f(d), self._log_jacobian_inv(d)


class ComovingDistanceConverter(DistanceConverter):
    """
    Object to convert luminosity distance with a prior that is uniform
    in comoving volume to a parameter with a uniform prior.

    Parameters
    ----------
    d_min, d_max : float
        Minimum and maximum distances
    units : str, optional
        Units used for the distance, must be compatible with astropy units.
    cosomology : str, optional
        Cosmology used for conversion, must be compatible with astropy.
    scale : float, optional
        Rescaling applied to distance after converting to comoving distance.
    pad : float, optional
        Padding used for min and max of interpolation range:
        min = (1 - pad) * d_min and max = (1 + pad) * d_max
    n_interp : int, optional
        Length of vector used for generating the look up table.
    """
    has_conversion = True
    has_jacobian = False

    def __init__(self, d_min=None, d_max=None, units='Mpc',
                 cosmology='Planck15', scale=1000.0, pad=0.05, n_interp=500):
        self.units = u.Unit(units)
        # TODO: this needs to update with bilby
        self.cosmology = cosmo.Planck15
        self.scale = np.float64(scale)
        self.pad = pad
        self.n_interp = n_interp

        self.dl_min = (1 - self.pad) * d_min
        self.dl_max = (1 + self.pad) * d_max

        logger.debug(f'Min and max distances: [{self.dl_min}, {self.dl_max}]')

        self.dc_min = self.cosmology.comoving_distance(cosmo.z_at_value(
            self.cosmology.luminosity_distance, self.dl_min * self.units)
            ).value
        self.dc_max = self.cosmology.comoving_distance(cosmo.z_at_value(
            self.cosmology.luminosity_distance, self.dl_max * self.units)
            ).value

        logger.debug('Making distance look up table')

        dc_array = np.linspace(self.dc_min, self.dc_max, self.n_interp)
        dl_array = self.cosmology.luminosity_distance(
            [cosmo.z_at_value(self.cosmology.comoving_distance, d * self.units)
                for d in dc_array]).value

        self.interp_dc2dl = interpolate.splrep(dc_array, dl_array)
        self.interp_dl2dc = interpolate.splrep(dl_array, dc_array)

    def to_uniform_parameter(self, dl):
        """Convert luminosity distance to a parameter with a uniform prior"""
        return ((interpolate.splev(dl, self.interp_dl2dc, ext=3) /
                 self.scale) ** 3.,
                np.zeros_like(dl))

    def from_uniform_parameter(self, dc):
        """Convert comoving distance to a parameter with a uniform prior"""
        return (interpolate.splev(self.scale * np.cbrt(dc),
                                  self.interp_dc2dl, ext=3),
                np.zeros_like(dc))


def get_distance_converter(prior):
    """Get a distance converter from the type of prior.

    If the prior is unknown a null converter is returned that has the
    identity rescaling.
    """
    if prior == 'uniform-comoving-volume':
        return ComovingDistanceConverter
    if prior == 'power-law':
        return PowerLawConverter
    else:
        logger.info(f'Prior {prior} is not known for distance')
        return NullDistanceConverter
