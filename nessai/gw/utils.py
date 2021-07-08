# -*- coding: utf-8 -*-
"""
Utilities specific to the gw subpackage.
"""
from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)

try:
    from astropy import cosmology as cosmo
    import astropy.units as u
except ImportError:
    logger.debug(
        'Could not import astropy, running with reduced functionality')


class DistanceConverter(ABC):
    """Base object for converting from a distance parameter to a uniform
    parameter.
    """
    has_conversion = False
    has_Jacobian = False

    @abstractmethod
    def to_uniform_parameter(self, d):
        raise NotImplementedError

    @abstractmethod
    def from_uniform_parameter(self, d):
        raise NotImplementedError


class NullDistanceConverter(DistanceConverter):
    """Converter that applies the identity transformation."""
    has_Jacobian = True

    def __init__(self, **kwargs):
        if kwargs:
            logger.warning(f'Kwargs {kwargs} will be ignored for distance')

    def to_uniform_parameter(self, d):
        return d, np.zeros_like(d)

    def from_uniform_parameter(self, d):
        return d, np.zeros_like(d)


class PowerLawConverter(DistanceConverter):
    """Convert from a distance parameter sample from a power law to a uniform
    parameter

    Assumes d proportional to :math:`d^{(\\text{power} + 1)}` following the \
        convention in Bilby.

    Parameters
    ----------
    power : float
        Power to use for the power-law.
    scale : float
        Factor used to rescale distance prior to converting to the uniform
        parameter.
    """
    has_conversion = True
    has_jacobian = True

    def __init__(self, power=None, scale=1000.0, **kwargs):
        if power is None:
            raise RuntimeError(
                'Must specify the power to use in the power-law'
            )
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
        Default is Planck15.
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
        try:
            self.cosmology = getattr(cosmo, cosmology)
            logger.info(f'Using cosmology: {cosmology}')
        except AttributeError:
            raise RuntimeError(
                f'Could not get specified cosmology ({cosmology}) from '
                '`astropy.cosmology`. Avaiable cosmologies are: '
                f'{cosmo.parameters.available}. See astropy documentation '
                'for more details.'
            )
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
        """Convert from a uniform parameter to luminosity distance.

        The uniform parameter is a scaled version of the comoving distance.
        """
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
