import os
import logging

import numpy as np
from scipy.stats import chi, norm

from ..proposal import FlowProposal
from ..flowmodel import FlowModel
from ..utils import (
    replace_in_list,
    rescale_zero_to_one,
    inverse_rescale_zero_to_one,
    rescale_minus_one_to_one,
    inverse_rescale_minus_one_to_one,
    detect_edge,
    determine_rescaled_bounds)
from .utils import (
    angle_to_cartesian,
    cartesian_to_angle,
    zero_one_to_cartesian,
    cartesian_to_zero_one,
    transform_from_precessing_parameters,
    transform_to_precessing_parameters,
    rescale_and_logit,
    rescale_and_sigmoid,
    ComovingDistanceConverter)
from .priors import (
    log_uniform_prior,
    log_2d_cartesian_prior,
    log_2d_cartesian_prior_sine,
    log_3d_cartesian_prior,
    log_spin_prior,
    log_spin_prior_uniform)

from .reparameterisations import get_gw_reparameterisation

logger = logging.getLogger(__name__)


class LegacyGWFlowProposal(FlowProposal):
    """
    A proposal specific to gravitational wave CBC
    """
    def __init__(self, model, reparameterisations={}, **kwargs):
        super().__init__(model, **kwargs)

        self.set_reparameterisations(reparameterisations)
        # list to itnernally track reparemeterisations
        self._reparameterisations = []
        self._search_angles = {}
        self._inversion = {}
        self._log_inversion = {}
        self._log_radial = {}
        self._angle_conversion = {}
        self._rescaled_min = {}
        self._rescaled_max = {}

        self._x_prime_prior_parameters = \
            ['mass_ratio_inv', 'chirp_mass_prime', 'time', 'dc3',
             'sky_x', 'sky_y', 'sky_z', 'psi_x', 'psi_y', 'iota_x', 'iota_y',
             'tilt_1_x', 'tilt_1_y', 'tilt_2_x', 'tilt_2_y', 'a_1_x', 'a_1_y',
             'a_2_x', 'a_2_y', 'phi_12_x', 'phi_12_y', 'phi_jl_x', 'phi_jl_y',
             'theta_jn_x', 'theta_jn_y',
             's1x', 's1y', 's1z', 's2x', 's2y', 's2z']

        self._default_inversion_parameters = ['mass_ratio', 'a_1', 'a_2',
                                              'luminosity_distance']

        self._default_angles = ['psi', 'phase', 'iota', 'theta_jn', 'dec',
                                'ra', 'tilt_1', 'tilt_2', 'cos_theta_jn',
                                'cos_tilt_1', 'cos_tilt_2', 'phi_12', 'phi_jl']

    def set_reparameterisations(self, reparameterisations):
        """
        Set the relevant reparamterisation flags
        """
        defaults = dict(mass_inversion=False,
                        reduced_quaternions=True, distance_rescaling=False,
                        norm_quaternions=False, rescale_angles=True,
                        euler_convention='ZYZ', angular_decomposition=True,
                        minus_one_to_one=True, log_inversion=False,
                        log_radial=False, inversion=True, exclude=[],
                        convert_to_angle=False, default_rescaling=[],
                        spin_conversion=False, spin_conversion_config={},
                        uniform_distance_parameter=False,
                        uniform_distance_parameter_config={},
                        spin_logit=False,
                        spin_logit_config={},
                        use_x_prime_prior=False)
        defaults.update(reparameterisations)

        if defaults['mass_inversion'] and defaults['inversion']:
            raise RuntimeError(
                'Mass inversion and inversion are not compatible')

        if (defaults['inversion'] is True and
                defaults['log_inversion'] is True):
            raise RuntimeError('To use inversion and log-inversion, specify '
                               'the parameters for each as list')
        if (isinstance(defaults['inversion'], list) and
                defaults['log_inversion'] is True):
            raise RuntimeError('Cannot set inversion to a list and use '
                               'default parameters for log inversion.')
        if (isinstance(defaults['log_inversion'], list) and
                defaults['inversion'] is True):
            raise RuntimeError('Cannot set log inversion to a list and use '
                               'default parameters for inversion.')

        if all([isinstance(inv, list) for inv in [defaults['inversion'],
                                                  defaults['log_inversion']]]):
            if s := (set(defaults['inversion'])
                     & set(defaults['log_inversion'])):
                raise RuntimeError('Inversion and log_inversion have common '
                                   f'parameters: {s}')

        logger.info('Reparameterisations:')
        for k, v in defaults.items():
            logger.info(f'{k}: {v}')
            setattr(self, k, v)

    def setup_angle(self, name, radial_name=False, scale=1.0, zero='bound'):
        """
        Add an angular parameter to the list of reparameterisations
        """
        if not radial_name:
            radial_name = name + '_radial'
            self.names.append(radial_name)
            self.rescaled_names.append(radial_name)

        x_name = name + '_x'
        y_name = name + '_y'
        replace_in_list(self.rescaled_names, [name, radial_name],
                        [x_name, y_name])

        self._search_angles[name] = {
            'angle': name, 'radial': radial_name,
            'x': x_name, 'y': y_name, 'scale': scale, 'zero': zero}

        logger.debug(f'Added {name} with config: {self._search_angles[name]}')

    @property
    def inversion_parameters(self):
        """
        Returns a list of parameters to which an inversion (normal or log)
        """
        parameters = []
        if isinstance(self.inversion, list):
            parameters += self.inversion
        if isinstance(self.log_inversion, list):
            parameters += self.log_inversion
        if not parameters and any([self.inversion, self.log_inversion]):
            parameters = self._default_inversion_parameters
        return parameters

    def add_inversion(self, name):
        """
        Setup inversion
        """
        rescaled_name = name + '_inv'
        replace_in_list(self.rescaled_names, [name],
                        [rescaled_name])

        # if name exists in the physical space, change it
        # else (e.g. radial parameters) leave as is
        self._inversion[name] = {
            'name': name, 'rescaled_name': rescaled_name,
            'rescale': True, 'invert': None,
            'min': self.model.bounds[name][0],
            'max': self.model.bounds[name][1]}

        logger.debug(f'Added {name} to parameters with inversion')

    def add_log_inversion(self, name):
        """
        Setup log inversion
        """
        # If upper bound is not 1, then rescale so require offset
        if not self.model.bounds[name][1] == 1 or \
                self.model.bounds[name][0] == 0:
            rescale = True
            offset = 0.0 * np.ptp(self.model.bounds[name])
        else:
            rescale = False
            offset = 0.0

        rescaled_name = name + '_inv'
        replace_in_list(self.rescaled_names, [name],
                        [rescaled_name])

        # if name exists in the physical space, change it
        # else (e.g. radial parameters) leave as is
        self._log_inversion[name] = {
            'name': name, 'rescaled_name': rescaled_name,
            'offset': offset, 'rescale': rescale, 'invert': None}

        logger.debug(f'Added {name} to parameters with log inversion')

    def add_angle_conversion(self, name, mode='split'):
        radial_name = name + '_radial'
        self.names.append(radial_name)
        self.rescaled_names.append(radial_name)

        x_name = name + '_x'
        y_name = name + '_y'
        replace_in_list(self.rescaled_names, [name, radial_name],
                        [x_name, y_name])

        self._angle_conversion[name] = {
            'name': name, 'radial': radial_name,
            'x': x_name, 'y': y_name, 'apply': True, 'mode': mode}

        logger.debug(f'{name} will be converted to an angle')

    def configure_time(self):
        """
        Configure the time parameter if present
        """
        time = [t for t in self.names if 'time' in t]
        if len(time) > 1:
            raise RuntimeError(f'Found more than one time: {time}')
        elif not len(time):
            self.time = False
        else:
            self.time = time[0]
            replace_in_list(self.rescaled_names, [self.time], ['time'])
            self._remaining.remove(self.time)
            # geocent time is handled different to other parameters,
            # we leave it in the defaults and change the prior bounds
            # we then only need to subtract the offset if it present
            # set offset as the midpoint of the prior
            # the bounds will then be +/- duration/2
            self.time_offset = self.model.bounds[self.time][0] \
                + np.ptp(self.model.bounds[self.time]) / 2
            logger.debug(f'Time offset: {self.time_offset}')
            # Save the bounds since we're using different bounds
            self.time_bounds = self.model.bounds[self.time] - self.time_offset

    def configure_sky(self):
        """
        Configure the sky parameters
        """
        if all(p in self.names for p in ['ra', 'dec']):
            from .utils import ra_dec_to_cartesian, cartesian_to_ra_dec
            self.sky_angles = ['ra', 'dec']
            self.sky_to_cartesian = ra_dec_to_cartesian
            self.cartesian_to_sky = cartesian_to_ra_dec
        elif all(p in self.names for p in ['azimuth', 'zenith']):
            from .utils import (azimuth_zenith_to_cartesian,
                                cartesian_to_azimuth_zenith)
            self.sky_angles = ['azimuth', 'zenith']
            self.sky_to_cartesian = azimuth_zenith_to_cartesian
            self.cartesian_to_sky = cartesian_to_azimuth_zenith
        elif any(p in self.names for p in
                 ['ra', 'dec', 'azimuth', 'zenith']):
            raise RuntimeError(
                'Cannot use angular decompoisiton with only'
                'one of the two sky angles')
        else:
            self.sky_angles = []

        if self.angular_decomposition and self.sky_angles:
            replace_in_list(self.rescaled_names,
                            self.sky_angles, ['sky_x', 'sky_y'])
            [self._remaining.remove(a) for a in self.sky_angles]
            if ('luminosity_distance' not in self.names or
                    'luminosity_distance' not in self._remaining or
                    'luminosity_distance' in self.default_rescaling):
                self.names.append('sky_radial')
                self.distance = 'sky_radial'
                self.rescaled_names.append('sky_z')
            else:
                self.distance = 'luminosity_distance'
                replace_in_list(self.rescaled_names, [self.distance],
                                ['sky_z'])
                self._remaining.remove('luminosty_disance')
            self._reparameterisations.append('sky')
            logger.info('Using angular decomposition of sky for: '
                        f'{self.sky_angles}')
        elif self.sky_angles:
            logger.warning(
                'Sampling sky but not using Cartesian reparmeterisation!')

    def configure_angles(self):
        """
        Configure angles
        """
        if self.angular_decomposition:
            if self.rescale_angles:
                if not isinstance(self.rescale_angles, list):
                    if isinstance(self.rescale_angles, bool):
                        self.rescale_angles = self._default_angles
                    elif isinstance(self.rescale_angles, str):
                        if self.rescale_angles == 'all':
                            self.rescale_angles = self._default_angles
                        else:
                            raise ValueError(
                                'Unknown value for rescale_angles: '
                                f'{self.rescale_angles}')
                logger.debug(f'Angles to rescale {self.rescale_angles}')
            else:
                self.rescale_angles = []

            logger.debug('Checking source angles')
            for a in ['psi', 'theta_jn', 'iota', 'phase', 'cos_theta_jn']:
                if a in self.names and a in self._remaining:
                    if a in self.rescale_angles:
                        scale = 2. * np.pi / np.ptp(self.model.bounds[a])
                    else:
                        scale = 1.0
                    if a in ['cos_theta_jn']:
                        zero = 'centre'
                    else:
                        zero = 'bound'
                    self.setup_angle(a, scale=scale, zero=zero)
                    self._remaining.remove(a)

            logger.debug('Checking spin angles')
            for i in [1, 2]:
                if ((((a := f'tilt_{i}') in self.names) or
                        ((a := f'cos_tilt_{i}') in self.names)) and
                        a in self._remaining):
                    if 'cos' in a:
                        zero = 'centre'
                    else:
                        zero = 'bound'
                    if a in self.rescale_angles:
                        scale = 2. * np.pi / np.ptp(self.model.bounds[a])
                    else:
                        scale = 1.0
                    if ((radial := f'a_{i}') in self.names and
                            radial in self._remaining and
                            not (radial in self._log_inversion or
                                 radial in self._inversion or
                                 radial in self._angle_conversion or
                                 radial in self.default_rescaling)):
                        self.setup_angle(a, radial, scale=scale, zero=zero)
                        self._remaining.remove(radial)
                    else:
                        self.setup_angle(a, scale=scale, zero=zero)
                        self._remaining.remove(a)

            for a in ['phi_jl', 'phi_12']:
                if a in self.names and a in self._remaining:
                    self.setup_angle(a, scale=1.0, zero='bound')
                    self._remaining.remove(a)
        else:
            logger.warning('Angles are not coverted to Cartesian!')

    def set_rescaling(self):
        """
        Set the rescaling functions
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()

        self._remaining = self.names.copy()
        [self._remaining.remove(p) for p in self.default_rescaling]

        self._min = {n: self.model.bounds[n][0] for n in self.model.names}
        self._max = {n: self.model.bounds[n][1] for n in self.model.names}
        if isinstance(self.inversion_type, str):
            if self.inversion_type not in ('split', 'duplicate', 'reflexion'):
                raise RuntimeError(
                        f'Unknown inversion type: {self.inversion_type}')
            if self.inversion is True:
                self.inversion_type = \
                    {p: self.inversion_type for
                     p in self._default_inversion_parameters}
            elif self.log_inversion is True:
                self.inversion_type = \
                    {p: self.inversion_type for
                     p in self._default_inversion_parameters}
            else:
                self.inversion_type = \
                    {p: self.inversion_type for
                     p in self.inversion_parameters}

        elif isinstance(self.inversion_type, dict) and self.inversion_type:
            if self.inversion is True:
                self.inversion = list(self.inversion_type.keys())
            elif self.log_inversion is True:
                self.log_inversion = list(self.inversion_type.keys())
            elif (not self.inversion and not self.log_inversion):
                raise RuntimeError('Inversion type specified as dict but '
                                   'inverison and log inversion are both '
                                   'disabled')
            elif (set(self.inversion_parameters)
                  - set(self.inversion_type.keys())):
                raise RuntimeError('If inversion is a list and inversion '
                                   'type is a dict, the entries in the '
                                   'list must match the keys!')
        if self.log_inversion:
            if isinstance(self.log_inversion, list):
                for p in self.log_inversion:
                    self.add_log_inversion(p)
                    self._remaining.remove(p)
            else:
                for p in ['mass_ratio', 'luminosity_distance', 'a_1', 'a_2']:
                    if p in self.names:
                        self.add_log_inversion(p)
                        self._remaining.remove(p)

        if self.inversion:
            logger.info(f'Inversion types: {self.inversion_type}')
            if isinstance(self.inversion, list):
                for p in self.inversion:
                    if p in self.names:
                        self.add_inversion(p)
                        self._remaining.remove(p)
                    elif not p == 'dc3':
                        logger.debug(f'Cannot apply inversion to {p}, '
                                     'parameter not being sampled')
            else:
                for p in self._default_inversion_parameters:
                    if p in self.names:
                        self.add_inversion(p)
                        self._remaining.remove(p)
                    else:
                        logger.debug(f'Cannot apply inversion to {p}, '
                                     'parameter not being sampled')

        if self.log_radial:
            log_radial = ['luminosity_distance', 'a_1', 'a_2']
            if not isinstance(self.log_radial, list):
                self._log_radial = [p for p in log_radial
                                    if p not in self._log_inversion]
            else:
                self._log_radial = [p for p in self.log_radial
                                    if p not in self._log_inversion]
            [self._remaining.remove(p) for p in self._log_radial]
            logger.debug(f'Using log radial for {self._log_radial}')

        if self.uniform_distance_parameter:
            self.setup_uniform_distance_parameter(
                **self.uniform_distance_parameter_config)

        if self.spin_conversion:
            self.configure_spin_conversion(**self.spin_conversion_config)

        if self.convert_to_angle:
            if isinstance(self.convert_to_angle, list):
                for p in self.convert_to_angle:
                    if p in self.names:
                        self.add_angle_conversion(p, mode='split')
                        self._remaining.remove(p)
                    else:
                        logger.debug(f'Cannot convert {p} to angle, '
                                     'parameter not being sampled')
            if isinstance(self.convert_to_angle, dict):
                for k, v in self.convert_to_angle.items():
                    if k in self.names:
                        self.add_angle_conversion(k, mode=v)
                        self._remaining.remove(k)
                    else:
                        logger.debug(f'Cannot convert {k} to angle, '
                                     'parameter not being sampled')

        else:
            self.convert_to_angle = []

        if self.spin_logit:
            self.setup_spin_logit(**self.spin_logit_config)

        self.configure_sky()

        self.configure_time()

        self.configure_angles()

        if self.mass_inversion:
            raise NotImplementedError()

        # Default -1 to 1 rescaling
        if self.minus_one_to_one:
            self.default_rescaling += list(
                (set(self.names) & set(self.rescaled_names))
                - set(self.exclude) - set(self.default_rescaling))
            replace_in_list(self.rescaled_names, self.default_rescaling,
                            [d + '_prime' for d in self.default_rescaling])

        self._rescale_factor = np.ptp(self.rescale_bounds)
        self._rescale_shift = self.rescale_bounds[0]

        if (not all(p in self._x_prime_prior_parameters
                    for p in self.rescaled_names)
                and self.use_x_prime_prior):
            raise RuntimeError(
                'x prime space includes parameters that are not included in x '
                f'prime priors.\n x prime parameters: {self.rescaled_names}\n'
                'x prime parameters in prior: ',
                self._x_prime_prior_parameters)

        self.rescale_parameters = 'all'
        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def check_state(self, x):
        """
        Check the state of the rescaling before training
        """
        if self._log_inversion:
            for c in self._log_inversion.values():
                c['invert'] = None
        if self._inversion:
            for c in self._inversion.values():
                c['invert'] = None
                if self.update_bounds:
                    c['min'] = np.min(x[c['name']])
                    c['max'] = np.max(x[c['name']])

        if self.uniform_distance_parameter:
            self._dc3_invert = None
            if self.update_bounds:
                self._dc3_min = \
                    self.convert_to_dc3(np.min(x['luminosity_distance']))
                self._dc3_max = \
                    self.convert_to_dc3(np.max(x['luminosity_distance']))

        if self.update_bounds:
            self._min = {n: np.min(x[n]) for n in self.model.names}
            self._max = {n: np.max(x[n]) for n in self.model.names}

        if self.use_x_prime_prior:
            self.update_rescaled_bounds()

    def update_rescaled_bounds(self, rescaled_names=None,
                               xmin=None, xmax=None):
        if rescaled_names is not None:
            for rn, mn, mx in zip(rescaled_names, xmin, xmax):
                self._rescaled_min[rn] = xmin
                self._rescaled_max[rn] = xmax

        else:
            for n, rn in zip(['chirp_mass', 'geocent_time'],
                             ['chirp_mass_prime', 'time']):
                if n in self.model.names:
                    self._rescaled_min[rn], _ = rescale_minus_one_to_one(
                        self.model.bounds[n][0], self._min[n], self._max[n])
                    self._rescaled_max[rn], _ = rescale_minus_one_to_one(
                        self.model.bounds[n][1], self._min[n], self._max[n])

    def setup_uniform_distance_parameter(self, scale_factor=1000, **kwargs):
        """
        Set up the uniform distance parameter dc3

        Parameters
        ----------
        scale_factor : float, (optional)
            Factor used to rescale comoving distance
        kwargs :
            Keyword arguments parsed to `ComovingDistanceConverter`
        """
        if 'luminosity_distance' not in self.names:
            raise RuntimeError('Uniform distance parameter is only compatible '
                               'with luminosity distance')
        if not self.use_x_prime_prior:
            raise RuntimeError('Cannot use dc3 without x prime prior')

        self.distance_converter = ComovingDistanceConverter(
            d_min=self.model.bounds['luminosity_distance'][0],
            d_max=self.model.bounds['luminosity_distance'][1],
            **kwargs)

        self._dc3_prior_min = \
            self.convert_to_dc3(self.model.bounds['luminosity_distance'][0])
        self._dc3_prior_max = \
            self.convert_to_dc3(self.model.bounds['luminosity_distance'][1])

        self._dc3_min = self._dc3_prior_min.copy()
        self._dc3_max = self._dc3_prior_max.copy()

        replace_in_list(self.rescaled_names, ['luminosity_distance'], ['dc3'])
        self._remaining.remove('luminosity_distance')

    def convert_to_dl(self, dc3):
        """
        Convert from uniform distance parameter dc3 to luminosity distance
        """
        return self.distance_converter.from_uniform_parameter(dc3)[0]

    def convert_to_dc3(self, dl):
        """
        Convert to uniform distance parameter dc3
        """
        return self.distance_converter.to_uniform_parameter(dl)[0]

    def setup_spin_logit(self, fuzz_factor=0.01):
        if 'a_1' in self.names and 'a_2' in self.names:
            self._spin_fuzz_factor = fuzz_factor
            self._remaining.remove('a_1')
            self._remaining.remove('a_2')
            replace_in_list(self.rescaled_names, ['a_1', 'a_2'],
                            ['a_1_logit', 'a_2_logit'])
        else:
            logger.debug('Missing spin magnitudes')

    def configure_spin_conversion(self, m1=20, m2=20, phase=0, f_ref=20,
                                  scale_factor=1, use_cbrt=True):
        self._m1 = float(m1)
        self._m2 = float(m2)
        self._phase = float(phase)
        self._f_ref = float(f_ref)
        self.scale_factor = scale_factor
        self._spin_use_cbrt = use_cbrt

        if self._spin_use_cbrt:
            self._spin_k1 = np.cbrt(self.model.bounds['a_1'][1])
            self._spin_k2 = np.cbrt(self.model.bounds['a_2'][1])
        else:
            self._spin_k1 = self.model.bounds['a_1'][1]
            self._spin_k2 = self.model.bounds['a_2'][1]

        self.precessing_params = \
            ['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2']
        self.cartesian_spin_params = \
            ['iota', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
        replace_in_list(self.rescaled_names,
                        self.precessing_params,
                        self.cartesian_spin_params)

        [self._remaining.remove(p) for p in self.precessing_params]

        self.names.append('iota_radial')
        self.rescaled_names.append('iota_radial')

        replace_in_list(self.rescaled_names, ['iota', 'iota_radial'],
                        ['iota_x', 'iota_y'])

        if 'iota' in self.rescale_angles:
            self._iota_scale = 2.0
        else:
            self._iota_scale = 1.0

    def _convert_spins_from_precessing(self, theta_jn, phi_jl, tilt_1,
                                       tilt_2, phi_12, a_1, a_2, m1=None,
                                       m2=None, f_ref=None, phase=None):
        if m1 is None:
            m1 = self._m1
        if m2 is None:
            m2 = self._m2
        if f_ref is None:
            f_ref = self._f_ref
        if phase is None:
            phase = self._phase

        log_J = 0

        if self._spin_use_cbrt:
            a_1 = np.cbrt(a_1)
            a_2 = np.cbrt(a_2)
            log_J += (-np.log(3) - 2 * np.log(a_1))
            log_J += (-np.log(3) - 2 * np.log(a_2))

        iota, s1x, s1y, s1z, s2x, s2y, s2z, lj = \
            transform_from_precessing_parameters(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
                m1, m2, f_ref, phase)

        log_J += lj

        s1x *= self.scale_factor
        s1y *= self.scale_factor
        s1z *= self.scale_factor
        s2x *= self.scale_factor
        s2y *= self.scale_factor
        s2z *= self.scale_factor

        return iota, s1x, s1y, s1z, s2x, s2y, s2z, log_J

    def _convert_spins_to_precessing(self, iota, s1x, s1y, s1z, s2x, s2y,
                                     s2z, m1=None, m2=None, f_ref=None,
                                     phase=None):
        if m1 is None:
            m1 = self._m1
        if m2 is None:
            m2 = self._m2
        if f_ref is None:
            f_ref = self._f_ref
        if phase is None:
            phase = self._phase

        s1x /= self.scale_factor
        s1y /= self.scale_factor
        s1z /= self.scale_factor
        s2x /= self.scale_factor
        s2y /= self.scale_factor
        s2z /= self.scale_factor
        log_J = 0

        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, lj = \
            transform_to_precessing_parameters(
                iota, s1x, s1y, s1z, s2x, s2y, s2z, m1, m2, f_ref, phase)
        log_J += lj

        if self._spin_use_cbrt:
            log_J += (np.log(3) + 2 * np.log(a_1))
            log_J += (np.log(3) + 2 * np.log(a_2))
            a_1 = a_1 ** 3
            a_2 = a_2 ** 3

        return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, log_J

    def _apply_inversion(self, x, x_prime, log_J, name, rescaled_name,
                         invert, rescale, compute_radius, inversion_type,
                         xmin=None, xmax=None, x_array=None):

        # Allow specifying an array
        # This allows for an inverison to applied after other rescaling
        if x_array is None:
            x_array = x[name]
        if rescale:
            x_prime[rescaled_name], lj = rescale_zero_to_one(
                x_array, xmin=xmin, xmax=xmax)
            log_J += lj
        else:
            x_prime[rescaled_name] = x_array

        if invert == 'upper':
            x_prime[rescaled_name] = \
                    1 - x_prime[rescaled_name]

        if invert == 'both':
            if compute_radius:
                if x_prime[rescaled_name][0] < 0.5:
                    lower = np.arange(x.size, 2 * x.size)
                    upper = np.array([], dtype=int)
                else:
                    lower = np.array([], dtype=int)
                    upper = np.arange(x.size, 2 * x.size)
                x_prime = np.tile(x_prime, 2)
                x = np.tile(x, 2)
                log_J = np.tile(log_J, 2)
            else:
                lower_samples = \
                    np.where(x_prime[rescaled_name] <= 0.5)[0]
                upper_samples = \
                    np.where(x_prime[rescaled_name] > 0.5)[0]
                lower = np.random.choice(lower_samples,
                                         lower_samples.size // 2,
                                         replace=False)
                upper = np.random.choice(upper_samples,
                                         upper_samples.size // 2,
                                         replace=False)
            x_prime[rescaled_name][lower] *= -1
            x_prime[rescaled_name][upper] = \
                2.0 - x_prime[rescaled_name][upper]

        else:
            if inversion_type == 'duplicate' or compute_radius:
                x_inv = x_prime.copy()
                x_inv[rescaled_name] *= -1
                x_prime = np.concatenate([x_prime, x_inv])
                x = np.concatenate([x,  x])
                log_J = np.concatenate([log_J, log_J])
            else:
                inv = np.random.choice(x_prime.size,
                                       x_prime.size // 2,
                                       replace=False)
                x_prime[rescaled_name][inv] *= -1

        return x, x_prime, log_J

    def _reverse_inversion(self, x, x_prime, log_J, name, rescaled_name,
                           invert, rescale, xmin=None, xmax=None):

        if invert == 'both':
            lower = x_prime[rescaled_name] < 0.
            upper = x_prime[rescaled_name] > 1.
            x[name] = x_prime[rescaled_name]
            x[name][lower] *= -1
            x[name][upper] = 2 - x[name][upper]

        else:
            inv = x_prime[rescaled_name] < 0.
            x[name][~inv] = x_prime[rescaled_name][~inv]
            x[name][inv] = -x_prime[rescaled_name][inv]

            if invert == 'upper':
                x[name] = 1 - x[name]

        if rescale:
            x[name], lj = \
                inverse_rescale_zero_to_one(x[name],
                                            xmin=xmin,
                                            xmax=xmax)
            log_J += lj

        return x, x_prime, log_J

    def rescale(self, x, compute_radius=False, test=None):
        """
        Rescale from the x space to the x prime space
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = np.zeros(x_prime.size)

        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']

        if x.size == 1:
            x = np.array([x], dtype=x.dtype)

        if self.default_rescaling:
            for n in self.default_rescaling:
                x_prime[n + '_prime'] = self._rescale_factor \
                             * ((x[n] - self._min[n])
                                / (self._max[n] - self._min[n])) \
                             + self._rescale_shift

                log_J += (-np.log(self._max[n] - self._min[n])
                          + np.log(self._rescale_factor))

        if self.exclude:
            for n in self.exclude:
                if n in x.dtype.names:
                    x_prime[n] = x[n]

        if self.spin_conversion:
            precessing_params = [x[p] for p in self.precessing_params]
            cart_params = \
                self._convert_spins_from_precessing(*precessing_params)
            # Skip iota (first)
            log_J += cart_params[-1]
            for n, p in zip(self.cartesian_spin_params[1:], cart_params[1:-1]):
                x_prime[n] = p
            x_prime['iota_x'], x_prime['iota_y'], lj = angle_to_cartesian(
                cart_params[0], scale=self._iota_scale)
            log_J += lj

        if self.spin_logit:
            for a in ['a_1', 'a_2']:
                x_prime[a + '_logit'], lj = rescale_and_logit(
                    x[a],
                    self.model.bounds[a][0] - self._spin_fuzz_factor,
                    self.model.bounds[a][1] + self._spin_fuzz_factor)
                log_J += lj

        if self._log_inversion:
            for c in self._log_inversion.values():
                if c['rescale']:
                    x_prime[c['rescaled_name']], lj = rescale_zero_to_one(
                        x[c['name']],
                        xmin=self.model.bounds[c['name']][0] - c['offset'],
                        xmax=self.model.bounds[c['name']][1])
                    log_J += lj

                if c['invert'] is None:
                    c['invert'] = detect_edge(
                        x_prime[c['rescaled_name']],
                        **self.detect_edges_kwargs)

                if c['invert'] == 'lower':
                    x_prime[c['rescaled_name']] = \
                            1 - x_prime[c['rescaled_name']]

                x_prime[c['rescaled_name']] = \
                    np.log(x_prime[c['rescaled_name']])

                log_J -= x_prime[c['rescaled_name']]

                if c['invert']:

                    if self.inversion_type[c['name']] == 'duplicate':
                        x_inv = x_prime.copy()
                        x_inv[c['rescaled_name']] *= -1
                        x_prime = np.concatenate([x_prime, x_inv])
                        x = np.concatenate([x,  x])
                        log_J = np.concatenate([log_J, log_J])
                    else:
                        inv = np.random.choice(x_prime.size, x_prime.size // 2,
                                               replace=False)
                        x_prime[c['rescaled_name']][inv] *= -1

        if self.uniform_distance_parameter:
            dc3 = self.convert_to_dc3(x['luminosity_distance'])
            # Edge detection
            if self._dc3_invert is None:
                if 'dc3' in self.inversion:
                    self._dc3_invert = detect_edge(
                            dc3,
                            x_range=[self._dc3_prior_min, self._dc3_prior_max],
                            allow_both=False,
                            allowed_bounds=['upper'],
                            test=test,
                            **self.detect_edges_kwargs)
                else:
                    self._dc3_invert = False

                xmin, xmax = determine_rescaled_bounds(
                        self._dc3_prior_min,
                        self._dc3_prior_max,
                        self._dc3_min,
                        self._dc3_max,
                        self._dc3_invert)
                self.update_rescaled_bounds(['dc3'], [xmin], [xmax])

            if self._dc3_invert:
                x, x_prime, log_J = self._apply_inversion(
                    x, x_prime, log_J, 'luminosity_distance', 'dc3',
                    self._dc3_invert, True, compute_radius,
                    self.inversion_type['dc3'], xmin=self._dc3_min,
                    xmax=self._dc3_max, x_array=dc3)

            else:
                x_prime['dc3'], lj = rescale_minus_one_to_one(
                    dc3, self._dc3_min, self._dc3_max)
                log_J += lj

        if self._inversion:
            for c in self._inversion.values():
                if c['invert'] is None:
                    if self.inversion_type[c['name']] == 'reflexion':
                        both = True
                    else:
                        both = False
                    c['invert'] = detect_edge(
                        x[c['name']],
                        allow_both=both,
                        test=test,
                        **self.detect_edges_kwargs)
                    logger.debug(f"Inversion for {c['name']}: {c['invert']}")
                    if self.use_x_prime_prior:
                        # Set the prior bounds in the x_prime space
                        xmin, xmax = determine_rescaled_bounds(
                            self.model.bounds[c['name']][0],
                            self.model.bounds[c['name']][1],
                            c['min'],
                            c['max'],
                            invert=c['invert'])
                        self.update_rescaled_bounds([c['rescaled_name']],
                                                    [xmin], [xmax])
                if c['invert']:
                    x, x_prime, log_J = self._apply_inversion(
                        x, x_prime, log_J, c['name'], c['rescaled_name'],
                        c['invert'], c['rescale'], compute_radius,
                        self.inversion_type[c['name']], xmin=c['min'],
                        xmax=c['max'])
                else:
                    if c['rescale']:
                        x_prime[c['rescaled_name']], lj = \
                            rescale_minus_one_to_one(x[c['name']],
                                                     xmin=c['min'],
                                                     xmax=c['max'])
                        log_J += lj
                    else:
                        x_prime[c['rescaled_name']] = x[c['name']].copy()

        if self._angle_conversion:
            for c in self._angle_conversion.values():
                p, lj = rescale_zero_to_one(
                    x[c['name']],
                    xmin=self.model.bounds[c['name']][0],
                    xmax=self.model.bounds[c['name']][1])
                log_J += lj
                # if computing the radius, set duplicate=True
                if ((c['mode'] == 'split' and compute_radius)
                        or c['mode'] == 'duplicate'):
                    x_prime = np.concatenate([x_prime, x_prime])
                    x = np.concatenate([x,  x])
                    log_J = np.concatenate([log_J, log_J])
                    x_prime[c['x']], x_prime[c['y']], lj = \
                        zero_one_to_cartesian(p, mode='duplicate')
                    log_J += lj

                else:
                    x_prime[c['x']], x_prime[c['y']], lj = \
                        zero_one_to_cartesian(p, mode=c['mode'])

                    log_J += lj

        if 'sky' in self._reparameterisations:
            if self.distance == 'luminosity_distance':
                if 'luminosity_distance' in self._log_radial:
                    r, lj = rescale_zero_to_one(
                        x[self.distance],
                        xmin=self.model.bounds[self.distance][0] - 0.1,
                        xmax=self.model.bounds[self.distance][1])
                    log_J += lj
                    r = -np.log(r)
                    # logJ = log(1/r) where r is the value before applying log
                    # have log(1/r) so use this
                    log_J += r
                else:
                    r, lj = rescale_zero_to_one(
                        x[self.distance],
                        xmin=self.model.bounds[self.distance][0],
                        xmax=self.model.bounds[self.distance][1])
                    log_J += lj
            else:
                r = None

            x_prime['sky_x'], x_prime['sky_y'], x_prime['sky_z'], lj = \
                self.sky_to_cartesian(x[self.sky_angles[0]],
                                      x[self.sky_angles[1]], r)
            log_J += lj

        if self.time:
            t = x[self.time] - self.time_offset
            x_prime['time'], lj = rescale_minus_one_to_one(
                t, self.time_bounds[0], self.time_bounds[1])
            log_J += lj

        if self._search_angles:
            for a in self._search_angles.values():
                # if the radial parameter is present in x
                # use it, else samples with be drawn from a chi with
                # 2 d.o.f
                if (n := a['radial']) in self.model.names:
                    r, lj = rescale_zero_to_one(
                        x[n], xmin=self._min[n], xmax=self._max[n])
                    log_J += lj
                    # r = np.log(r)
                    # log_J -= r
                    # r = 1 - r
                    # r = - np.log(1 - x[n])
                    # log_J += np.positive(r)   # log|J| = np.log(1-r)
                else:
                    r = None
                x_prime[a['x']], x_prime[a['y']], lj = angle_to_cartesian(
                    x[a['angle']], r=r, scale=a['scale'])

                log_J += lj

        if self.mass_inversion:
            raise NotImplementedError

        return x_prime, log_J

    def inverse_rescale(self, x_prime):
        """
        Rescale from the x prime  space to the x space
        """
        x = np.zeros([x_prime.size], dtype=self.x_dtype)
        log_J = np.zeros(x_prime.size)

        x['logP'] = x_prime['logP']
        x['logL'] = x_prime['logL']

        # Sort mass ratio first so that phase, tilt angles and magntiude
        # are correct before applying other rescaling
        if self.mass_inversion:
            raise NotImplementedError

        if 'sky' in self._reparameterisations:
            x[self.sky_angles[0]], x[self.sky_angles[1]], r, lj = \
                self.cartesian_to_sky(x_prime['sky_x'], x_prime['sky_y'],
                                      x_prime['sky_z'])
            log_J += lj

            if self.distance == 'luminosity_distance':
                if 'luminosity_distance' in self._log_radial:
                    log_J -= r.copy()
                    r = np.exp(-r)
                    r, lj = inverse_rescale_zero_to_one(
                        r, xmin=self.model.bounds[self.distance][0] - 0.1,
                        xmax=self.model.bounds[self.distance][1])
                    log_J += lj
                else:
                    r, lj = inverse_rescale_zero_to_one(
                        r, xmin=self.model.bounds[self.distance][0],
                        xmax=self.model.bounds[self.distance][1])
                    log_J += lj
            x[self.distance] = r

        if self.time:
            t, lj = inverse_rescale_minus_one_to_one(x_prime['time'],
                                                     self.time_bounds[0],
                                                     self.time_bounds[1])
            # This will break with casting rules
            x[self.time] = np.float64(t) + np.float64(self.time_offset)
            log_J += lj

        if self._search_angles:
            for a in self._search_angles.values():
                x[a['angle']], r, lj = cartesian_to_angle(
                    x_prime[a['x']], x_prime[a['y']], scale=a['scale'],
                    zero=a['zero'])
                log_J += lj
                # if the radial parameter is defined in the model
                # rescale it using the bounds
                if (n := a['radial']) in self.model.names:
                    r, lj = inverse_rescale_zero_to_one(
                        r, xmin=self._min[n], xmax=self._max[n])
                    log_J += lj
                x[a['radial']] = r

        if self._angle_conversion:
            for c in self._angle_conversion.values():
                p, r, lj = cartesian_to_zero_one(
                    x_prime[c['x']], x_prime[c['y']])
                log_J += lj

                x[c['name']], lj = inverse_rescale_zero_to_one(
                    p, xmin=self.model.bounds[c['name']][0],
                    xmax=self.model.bounds[c['name']][1])
                log_J += lj
                x[c['radial']] = r

        if self._log_inversion:
            for c in self._log_inversion.values():
                inv = x_prime[c['rescaled_name']] > 0.
                x[c['name']][~inv] = np.exp(x_prime[c['rescaled_name']][~inv])
                x[c['name']][inv] = np.exp(-x_prime[c['rescaled_name']][inv])

                if c['invert'] == 'lower':
                    x[c['name']] = 1 - x[c['name']]

                # for q_inv < 0 conversion is
                # exp(q_inv) for q_inv > 0 exp(-q_inv)
                # so Jacobian is log(exp(+/-q_inv))
                # i.e. q_inv and - q_inv respectively
                log_J[~inv] += x_prime[c['rescaled_name']][~inv]
                log_J[inv] -= x_prime[c['rescaled_name']][inv]

                if c['rescale']:
                    x[c['name']], lj = inverse_rescale_zero_to_one(
                            x[c['name']],
                            xmin=self.model.bounds[c['name']][0] - c['offset'],
                            xmax=self.model.bounds[c['name']][1])
                    log_J += lj

        if self._inversion:
            for c in self._inversion.values():
                if c['invert']:
                    x, x_prime, log_J = self._reverse_inversion(
                            x, x_prime, log_J, c['name'], c['rescaled_name'],
                            c['invert'], c['rescale'],
                            xmin=c['min'], xmax=c['max'])
                else:
                    if c['rescale']:
                        x[c['name']], lj = inverse_rescale_minus_one_to_one(
                            x_prime[c['rescaled_name']],
                            xmin=c['min'], xmax=c['max'])
                        log_J += lj
                    else:
                        x[c['name']] = x_prime[c['rescaled_name']].copy()

        if self.spin_conversion:
            iota, x['iota_radial'], lj = cartesian_to_angle(
                x_prime['iota_x'], x_prime['iota_y'], scale=self._iota_scale)
            log_J += lj
            # iota is not present in x_prime
            cart_params = [x_prime[p] for p in self.cartesian_spin_params[1:]]
            precessing_params = \
                self._convert_spins_to_precessing(iota, *cart_params)
            for n, p in zip(self.precessing_params, precessing_params[:-1]):
                x[n] = p
            log_J += precessing_params[-1]

        if self.spin_logit:
            for a in ['a_1', 'a_2']:
                x[a], lj = rescale_and_sigmoid(
                    x_prime[a + '_logit'],
                    self.model.bounds[a][0] - self._spin_fuzz_factor,
                    self.model.bounds[a][1] + self._spin_fuzz_factor)

                log_J += lj

        if self.uniform_distance_parameter:
            if self._dc3_invert:
                x, x_prime, log_J = self._reverse_inversion(
                        x, x_prime, log_J, 'luminosity_distance', 'dc3',
                        self._dc3_invert, True,
                        xmin=self._dc3_min, xmax=self._dc3_max)
            else:
                x['luminosity_distance'], lj = \
                    inverse_rescale_minus_one_to_one(x_prime['dc3'],
                                                     xmin=self._dc3_min,
                                                     xmax=self._dc3_max)
                log_J += lj
            # Another step is requires to covert back to dL
            x['luminosity_distance'] = \
                self.convert_to_dl(x['luminosity_distance'])

        if self.default_rescaling:
            for n in self.default_rescaling:
                x[n] = (self._max[n] - self._min[n]) \
                    * (x_prime[n + '_prime'] - self._rescale_shift) \
                    / self._rescale_factor + self._min[n]

                log_J += (np.log(self._max[n] - self._min[n])
                          - np.log(self._rescale_factor))

        if self.exclude:
            for n in self.exclude:
                if n in x_prime.dtype.names:
                    x[n] = x_prime[n]

        return x, log_J

    def log_prior(self, x):
        """
        Modified log prior that handles radial parameters
        """
        log_p = self.model.log_prior(x[self.model.names])

        if 'sky' in self._reparameterisations:
            if self.distance == 'sky_radial':
                log_p += chi.logpdf(x[self.distance], 3)
        if self._search_angles:
            for a in self._search_angles.values():
                if not (n := a['radial']) in self.model.names:
                    log_p += chi.logpdf(x[n], 2)

        if self._angle_conversion:
            for a in self._angle_conversion.values():
                log_p += chi.logpdf(x[a['radial']], 2)

        if self.spin_conversion and 'iota_radial' in self.names:
            log_p += chi.logpdf(x['iota_radial'], 2)

        return log_p

    def compute_rescaled_bounds(self, name):
        xmin = ((self.model.bounds[name][0] - self._min[name])
                / (self._max[name] - self._min[name]))

        xmin = min(xmin, 0)

        xmax = ((self.model.bounds[name][1] - self._min[name])
                / (self._max[name] - self._min[name]))

        xmax = max(xmax, 1)
        if name not in self.inversion or not self._inversion[name]['invert']:
            return 2 * xmin - 1, 2 * xmax - 1
        elif self._inversion[name]['invert'] == 'upper':
            return xmin - 1, 1 - xmin
        elif self._inversion[name]['invert'] == 'lower':
            return -xmax, xmax
        else:
            raise NotImplementedError

    def x_prime_log_prior(self, x_prime):
        """
        Priors redefined in the x_prime space

        Priors
        ------
        * Chirp mass and mass ratio: uniform (i.e. constant)
        * Time: uniform
        * polarsiation: uniform on pi (includes radial)
        """
        log_p = 0
        log_J = 0
        for n in ['chirp_mass_prime', 'mass_ratio_inv', 'time', 'dc3']:
            if n in self.rescaled_names:
                log_p += log_uniform_prior(x_prime[n],
                                           xmin=self._rescaled_min[n],
                                           xmax=self._rescaled_max[n])
        # Sky
        if 'sky_x' in self.rescaled_names:
            log_p += log_3d_cartesian_prior(x_prime['sky_x'], x_prime['sky_y'],
                                            x_prime['sky_z'])

        if 'psi_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior(
                x_prime['psi_x'], x_prime['psi_y'],
                k=self._search_angles['psi']['scale'] * np.pi)

        if 'phi_12_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior(
                x_prime['phi_12_x'], x_prime['phi_12_y'],
                k=self._search_angles['phi_12']['scale'] * np.pi)

        if 'phi_jl_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior(
                x_prime['phi_jl_x'], x_prime['phi_jl_y'],
                k=self._search_angles['phi_jl']['scale'] * np.pi)

        if 'a_1_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior(
                x_prime['a_1_x'], x_prime['a_1_y'],
                k=0.99)

        if 'a_2_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior(
                x_prime['a_2_x'], x_prime['a_2_y'],
                k=0.99)

        if 'iota_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior_sine(
                x_prime['iota_x'], x_prime['iota_y'])

        if 'theta_jn_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior_sine(
                x_prime['theta_jn_x'], x_prime['theta_jn_y'])

        if 'tilt_1_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior_sine(
                x_prime['tilt_1_x'], x_prime['tilt_1_y'])

        if 'tilt_2_x' in self.rescaled_names:
            log_p += log_2d_cartesian_prior_sine(
                x_prime['tilt_2_x'], x_prime['tilt_2_y'])

        if 's1x' in self.rescaled_names:
            if self._spin_use_cbrt:
                log_p += log_spin_prior_uniform(
                    x_prime['s1x'] / self.scale_factor,
                    x_prime['s1y'] / self.scale_factor,
                    x_prime['s1z'] / self.scale_factor,
                    x_prime['s2x'] / self.scale_factor,
                    x_prime['s2y'] / self.scale_factor,
                    x_prime['s2z'] / self.scale_factor,
                    k1=self._spin_k1,
                    k2=self._spin_k2)

            else:
                log_p += log_spin_prior(
                    x_prime['s1x'] / self.scale_factor,
                    x_prime['s1y'] / self.scale_factor,
                    x_prime['s1z'] / self.scale_factor,
                    x_prime['s2x'] / self.scale_factor,
                    x_prime['s2y'] / self.scale_factor,
                    x_prime['s2z'] / self.scale_factor,
                    k1=self._spin_k1,
                    k2=self._spin_k2)
        return log_p - log_J


class AugmentedGWFlowProposal(LegacyGWFlowProposal):

    def __init__(self, model, augment_features=1, **kwargs):
        super().__init__(model, **kwargs)
        self.augment_features = augment_features

    def set_rescaling(self):
        super().set_rescaling()
        self.augment_names = [f'e_{i}' for i in range(self.augment_features)]
        self.names += self.augment_names
        self.rescaled_names += self.augment_names
        self.exclude += self.augment_names
        logger.info(f'augmented x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(
            f'augmented x prime space parameters: {self.rescaled_names}'
        )

    def initialise(self):
        """
        Initialise the proposal class
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.set_rescaling()

        m = np.ones(self.rescaled_dims)
        m[-self.augment_features:] = -1
        self.flow_config['model_config']['kwargs']['mask'] = m

        self.flow_config['model_config']['n_inputs'] = self.rescaled_dims

        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def rescale(self, x):
        x_prime, log_J = super().rescale(x)
        if x_prime.size == 1:
            x_prime[self.augment_names] = self.augment_features * (0.,)
        else:
            for an in self.augment_names:
                x_prime[an] = np.random.randn(x_prime.size)
        return x_prime, log_J

    def augmented_prior(self, x):
        """
        Log guassian for augmented variables
        """
        logP = 0.
        for n in self.augment_names:
            logP += norm.logpdf(x[n])
        return logP

    def log_prior(self, x):
        """
        Compute the prior probability
        """
        logP = super().log_prior(x)
        logP += self.augmented_prior(x)
        return logP


class GWFlowProposal(FlowProposal):
    """Wrapper for FlowProposal that has defaults for CBC-PE"""
    aliases = {
        'chirp_mass': ('mass', None),
        'mass_ratio': ('mass_ratio', None),
        'ra': ('sky-ra-dec', ['DEC', 'dec', 'Dec']),
        'dec': ('sky-ra-dec', ['RA', 'ra']),
        'azimuth': ('sky-az-zen', ['zenith', 'zen', 'Zen', 'Zenith']),
        'zenith': ('sky-az-zen', ['azimuth', 'az', 'Az', 'Azimuth']),
        'theta_1': ('angle-sine', None),
        'theta_2': ('angle-sine', None),
        'tilt_1': ('angle-sine', None),
        'tilt_2': ('angle-sine', None),
        'theta_jn': ('angle-sine', None),
        'iota': ('angle-sine', None),
        'phi_jl': ('angle-2pi', None),
        'phi_12': ('angle-2pi', None),
        'phase': ('angle-2pi', None),
        'psi': ('angle-pi', None),
        'geocent_time': ('time', None),
        'a_1': ('to-cartesian', None),
        'a_2': ('to-cartesian', None),
        'luminosity_distance': ('distance', None),
    }

    def get_reparameterisation(self, reparameterisation):
        """Function to get reparameterisations that checks GW defaults and
        aliases
        """
        return get_gw_reparameterisation(reparameterisation)

    def add_default_reparameterisations(self):
        """
        Add default reparameterisations for parameters that have not been
        specified.
        """
        parameters = list(set(self.names)
                          - set(self._reparameterisation.parameters))
        logger.debug(f'Adding default reparameterisations for {parameters}')

        for p in parameters:
            logger.debug(f'Trying to add reparameterisation for {p}')
            if p in self._reparameterisation.parameters:
                logger.debug(f'Parameter {p} is already included')
                continue
            name, extra_params = self.aliases.get(p.lower(), (None, None))
            if name is None:
                logger.debug(f'{p} is not a known GW parameter')
                continue
            if extra_params is not None:
                p = [p] + [ep for ep in extra_params if ep in self.model.names]
            else:
                p = [p]
            prior_bounds = {k: self.model.bounds[k] for k in p}
            reparam, kwargs = get_gw_reparameterisation(name)
            logger.debug(
                f'Add reparameterisation for {p} with config: {kwargs}')
            self._reparameterisation.add_reparameterisation(
                reparam(parameters=p, prior_bounds=prior_bounds, **kwargs))
