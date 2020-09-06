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
    detect_edge)
from .utils import (
    angle_to_cartesian,
    cartesian_to_angle,
    sky_to_cartesian,
    cartesian_to_sky,
    zero_one_to_cartesian,
    cartesian_to_zero_one)

logger = logging.getLogger(__name__)


class GWFlowProposal(FlowProposal):
    """
    A proposal specific to gravitational wave CBC
    """
    def __init__(self, model, reparameterisations={}, **kwargs):
        super(GWFlowProposal, self).__init__(model, **kwargs)

        self.set_reparameterisations(reparameterisations)
        # list to itnernally track reparemeterisations
        self._reparameterisations = []
        self._search_angles = {}
        self._inversion = {}
        self._log_inversion = {}
        self._log_radial = {}
        self._angle_conversion = {}

        self._default_inversion_parameters = ['mass_ratio', 'a_1', 'a_2',
                                              'luminosity_distance']

    def set_reparameterisations(self, reparameterisations):
        """
        Set the relevant reparamterisation flags
        """
        defaults = dict(mass_inversion=False, flip=False,
                        reduced_quaternions=True, distance_rescaling=False,
                        norm_quaternions=False, rescale_angles=True,
                        euler_convention='ZYZ', angular_decomposition=True,
                        minus_one_to_one=True, log_inversion=False,
                        log_radial=False, inversion=True, exclude=[],
                        convert_to_angle=False, default_rescaling=[])
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

    def setup_angle(self, name, radial_name=False, scale=1.0):
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
            'x': x_name, 'y': y_name, 'scale': scale}

        logger.debug(f'Added {name} with config: {self._search_angles[name]}')

    @property
    def inversion_parameters(self):
        """
        Returns a list of parameters to which an inversion (normal or log)
        is applied
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
            'rescale': True, 'flip': None,
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
            'offset': offset, 'rescale': rescale, 'flip': None}

        logger.debug(f'Added {name} to parameters with log inversion')

    def add_angle_conversion(self, name):
        radial_name = name + '_radial'
        self.names.append(radial_name)
        self.rescaled_names.append(radial_name)

        x_name = name + '_x'
        y_name = name + '_y'
        replace_in_list(self.rescaled_names, [name, radial_name],
                        [x_name, y_name])

        self._angle_conversion[name] = {
            'name': name, 'radial': radial_name,
            'x': x_name, 'y': y_name, 'apply': True}

        logger.debug(f'{name} will be converted to an angle')

    def set_rescaling(self):
        """
        Set the rescaling functions
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()

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

        elif isinstance(self.inversion_type, dict):
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
            else:
                for p in ['mass_ratio', 'luminosity_distance', 'a_1', 'a_2']:
                    if p in self.names:
                        self.add_log_inversion(p)

        if self.inversion:
            if isinstance(self.inversion, list):
                for p in self.inversion:
                    self.add_inversion(p)
            else:
                for p in self._default_inversion_parameters:
                    if p in self.names:
                        self.add_inversion(p)
            logger.info(f'Inversion types: {self.inversion_type}')

        if self.log_radial:
            log_radial = ['luminosity_distance', 'a_1', 'a_2']
            if not isinstance(self.log_radial, list):
                self._log_radial = [p for p in log_radial
                                    if p not in self._log_inversion]
            else:
                self._log_radial = [p for p in self.log_radial
                                    if p not in self._log_inversion]
            logger.debug(f'Using log radial for {self._log_radial}')

        if self.convert_to_angle:
            if isinstance(self.convert_to_angle, list):
                for p in self.convert_to_angle:
                    self.add_angle_conversion(p)

        if self.angular_decomposition:
            if all(p in self.names for p in ['ra', 'dec']):
                replace_in_list(self.rescaled_names,
                                ['ra', 'dec'], ['sky_x', 'sky_y'])
                if 'luminosity_distance' not in self.names or \
                        'luminosity_distance' in self._log_inversion or \
                        'luminosity_distance' in self._inversion or \
                        'luminosity_distance' in self.convert_to_angle:
                    self.names.append('sky_radial')
                    self.distance = 'sky_radial'
                    self.rescaled_names.append('sky_z')
                else:
                    self.distance = 'luminosity_distance'
                    replace_in_list(self.rescaled_names, [self.distance],
                                    ['sky_z'])
                self._reparameterisations.append('sky')

            elif any(p in self.names for p in ['ra', 'dec']):
                raise RuntimeError(
                    'Cannot use angular decompoisiton with only'
                    'one of the two sky angles')

        if 'geocent_time' in self.names:
            self.time = 'geocent_time'
            replace_in_list(self.rescaled_names, [self.time], ['time'])
            # TODO: add catch for other time
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
            self._reparameterisations.append('time')

        if self.angular_decomposition:
            logger.debug('Checking source angles')
            for a in ['psi', 'theta_jn', 'iota', 'phase']:
                if a in self.names:
                    if self.rescale_angles:
                        scale = 2. * np.pi / np.ptp(self.model.bounds[a])
                    else:
                        scale = 1.0
                    self.setup_angle(a, scale=scale)
            logger.debug('Checking spin angles')
            for i in [1, 2]:
                if (a := f'tilt_{i}') in self.names:
                    if self.rescale_angles:
                        scale = 2. * np.pi / np.ptp(self.model.bounds[a])
                    else:
                        scale = 1.0
                    if ((radial := f'a_{i}') in self.names and
                            not (radial in self._log_inversion or
                                 radial in self._inversion or
                                 radial in self._angle_conversion or
                                 radial in self.default_rescaling)):
                        self.setup_angle(a, radial, scale=scale)
                    else:
                        self.setup_angle(a, scale=scale)

            for a in ['phi_jl', 'phi_12']:
                if a in self.names:
                    self.setup_angle(a, scale=1.0)

        if self.mass_inversion:
            if 'mass_ratio' in self.names:
                self._reparameterisations.append('mass_inversion')
                replace_in_list(self.rescaled_names, ['mass_ratio'],
                                ['mass_ratio_inv'])
            elif all(m in self.names for m in ['mass_1', 'mass_2']):
                self._reparameterisations.append('component_masses')
                replace_in_list(self.rescaled_names, ['mass_1', 'mass_2'],
                                ['mass_1_dbl', 'mass_2_dbl'])
            else:
                # Disable mass inversion
                self.mass_inversion = False

        # Default -1 to 1 rescaling
        if self.minus_one_to_one:
            self.default_rescaling += list(
                (set(self.names) & set(self.rescaled_names))
                - set(self.exclude) - set(self.default_rescaling))
            replace_in_list(self.rescaled_names, self.default_rescaling,
                            [d + '_prime' for d in self.default_rescaling])

        self._rescale_factor = np.ptp(self.rescale_bounds)
        self._rescale_shift = self.rescale_bounds[0]

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
                c['flip'] = None
        if self._inversion:
            for c in self._inversion.values():
                c['flip'] = None
                if self.update_bounds:
                    c['min'] = np.min(x[c['name']])
                    c['max'] = np.max(x[c['name']])

        if self.update_bounds:
            self._min = {n: np.min(x[n]) for n in self.model.names}
            self._max = {n: np.max(x[n]) for n in self.model.names}

    def rescale(self, x):
        """
        Rescale from the x space to the x prime space
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = np.zeros(x_prime.size)

        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']

        # If population the proposal and using split then force points
        # to be duplicate so maximum raidus can be used
        if x.size == 1:
            pop = True
            x = np.array([x], dtype=x.dtype)
        else:
            pop = False

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

        if self._log_inversion:
            for c in self._log_inversion.values():
                if c['rescale']:
                    x_prime[c['rescaled_name']], lj = rescale_zero_to_one(
                        x[c['name']],
                        xmin=self.model.bounds[c['name']][0] - c['offset'],
                        xmax=self.model.bounds[c['name']][1])
                    log_J += lj

                if c['flip'] is None:
                    c['flip'] = detect_edge(
                        x_prime[c['rescaled_name']],
                        bounds=[0, 1],
                        cutoff=self._edge_cutoff,
                        mode_range=self._edge_mode_range)
                    logger.debug(f"Log inversion for {c['name']}: {c['flip']}")

                if c['flip'] == 'lower':
                    x_prime[c['rescaled_name']] = \
                            1 - x_prime[c['rescaled_name']]

                x_prime[c['rescaled_name']] = \
                    np.log(x_prime[c['rescaled_name']])

                log_J -= x_prime[c['rescaled_name']]

                if c['flip']:

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

        if self._inversion:
            for c in self._inversion.values():
                if c['flip'] is None:
                    if self.inversion_type[c['name']] == 'reflexion':
                        both = True
                    else:
                        both = False
                    c['flip'] = detect_edge(
                        x[c['name']],
                        bounds=[c['min'], c['max']],
                        cutoff=self._edge_cutoff,
                        allow_none=self._allow_none,
                        both=both)
                    logger.debug(f"Inversion for {c['name']}: {c['flip']}")

                if c['flip']:

                    if c['rescale']:
                        x_prime[c['rescaled_name']], lj = rescale_zero_to_one(
                            x[c['name']], xmin=c['min'], xmax=c['max'])
                        log_J += lj

                    if c['flip'] == 'upper':
                        x_prime[c['rescaled_name']] = \
                                1 - x_prime[c['rescaled_name']]

                    if c['flip'] == 'both':
                        if pop:
                            if x_prime[c['rescaled_name']][0] < 0.5:
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
                                np.where(x_prime[c['rescaled_name']] <= 0.5)[0]
                            upper_samples = \
                                np.where(x_prime[c['rescaled_name']] > 0.5)[0]
                            lower = np.random.choice(lower_samples,
                                                     lower_samples.size // 2,
                                                     replace=False)
                            upper = np.random.choice(upper_samples,
                                                     upper_samples.size // 2,
                                                     replace=False)
                        x_prime[c['rescaled_name']][lower] *= -1
                        x_prime[c['rescaled_name']][upper] = \
                            2.0 - x_prime[c['rescaled_name']][upper]

                    else:
                        if (self.inversion_type[c['name']] == 'duplicate' or
                                pop):
                            x_inv = x_prime.copy()
                            x_inv[c['rescaled_name']] *= -1
                            x_prime = np.concatenate([x_prime, x_inv])
                            x = np.concatenate([x,  x])
                            log_J = np.concatenate([log_J, log_J])
                        else:
                            inv = np.random.choice(x_prime.size,
                                                   x_prime.size // 2,
                                                   replace=False)
                            x_prime[c['rescaled_name']][inv] *= -1
                else:
                    if c['rescale']:
                        x_prime[c['rescaled_name']], lj = \
                            rescale_minus_one_to_one(x[c['name']],
                                                     xmin=c['min'],
                                                     xmax=c['max'])
                        log_J += lj

        if self._angle_conversion:
            for c in self._angle_conversion.values():
                p, lj = rescale_zero_to_one(
                    x[c['name']],
                    xmin=self.model.bounds[c['name']][0],
                    xmax=self.model.bounds[c['name']][1])
                log_J += lj

                x_prime[c['x']], x_prime[c['y']], lj = \
                    zero_one_to_cartesian(p)

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
                sky_to_cartesian(x['ra'], x['dec'], r)
            log_J += lj

        if 'time' in self._reparameterisations:
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
                    r, lj = rescale_zero_to_one(x[n],
                                                xmin=self._min[n],
                                                xmax=self._max[n])
                    log_J += lj
                else:
                    r = None
                x_prime[a['x']], x_prime[a['y']], lj = angle_to_cartesian(
                    x[a['angle']], r=r, scale=a['scale'])

                log_J += lj

        if self.mass_inversion:
            if 'mass_ratio' in self.names:
                x_prime['mass_ratio_inv'] = np.log(x['mass_ratio'])
                x_prime_inv = x_prime.copy()
                x_prime_inv['mass_ratio_inv'] *= -1
                log_J -= np.log(x['mass_ratio'])

            elif 'component_masses' in self._reparameterisations:
                x['mass_1'], lj = rescale_minus_one_to_one(
                    x['mass_1'],
                    xmin=self.model.bounds['mass_1'][0],
                    xmax=self.model.bounds['mass_1'][1])
                log_J += lj
                x['mass_2'], lj = rescale_minus_one_to_one(
                    x['mass_2'],
                    xmin=self.model.bounds['mass_2'][0],
                    xmax=self.model.bounds['mass_2'][1])
                log_J += lj
                x_prime_inv = x_prime.copy()
                x_prime[['mass_1_dbl', 'mass_2_dbl']] = \
                    x[['mass_1', 'mass_2']]
                x_prime_inv[['mass_1_dbl', 'mass_2_dbl']] = \
                    x[['mass_2', 'mass_1']]

            # flip the phase
            if 'phase' in self._search_angles:
                # s = self._search_angles['phase']
                x_prime_inv[a['x']] *= -1
                x_prime_inv[a['y']] *= -1

            if all(t in self._search_angles for t in ['tilt_1', 'tilt_2']):
                t1 = self._search_angles['tilt_1']
                t2 = self._search_angles['tilt_2']
                x_prime_inv[[t1['x'], t1['y']]] = \
                    x_prime[[t2['x'], t2['y']]].copy()
                x_prime_inv[[t2['x'], t2['y']]] = \
                    x_prime[[t1['x'], t1['y']]].copy()

            elif any(t in self._search_angles for t in ['tilt_1', 'tilt_2']):
                raise RuntimeError(
                    'Cannot use q-inversion with only one tilt angle')

            x_prime = np.concatenate([x_prime, x_prime_inv])
            # Absolute value means jacobian is the same for either
            log_J = np.concatenate([log_J, log_J])

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
            if 'mass_ratio' in self.names:
                # Find `inverted` part
                inv = x_prime['mass_ratio_inv'] > 0.
                x['mass_ratio'][~inv] = \
                    np.exp(x_prime['mass_ratio_inv'][~inv])
                x['mass_ratio'][inv] = \
                    np.exp(-x_prime['mass_ratio_inv'][inv])
                # for q_inv < 0 conversion is exp(q_inv)
                # for q_inv > 0 exp(-q_inv)
                # so Jacobian is log(exp(+/-q_inv))
                # i.e. q_inv and - q_inv respectively
                log_J[~inv] += x_prime['mass_ratio_inv'][~inv]
                log_J[inv] -= x_prime['mass_ratio_inv'][inv]

            elif 'component_masses' in self._reparameterisations:
                inv = x_prime['mass_1_dbl'] < x_prime['mass_2_dbl']
                x_prime[['mass_1_dbl', 'mass_2_dbl']][inv] = \
                    x_prime[['mass_2_dbl', 'mass_1_dbl']][inv]
                x['mass_1'], lj = inverse_rescale_minus_one_to_one(
                    x_prime['mass_1_dbl'],
                    xmin=self.model.bounds['mass_1'][0],
                    xmax=self.model.bounds['mass_1'][1])
                log_J += lj
                x['mass_2'], lj = inverse_rescale_minus_one_to_one(
                    x_prime['mass_2_dbl'],
                    xmin=self.model.bounds['mass_2'][0],
                    xmax=self.model.bounds['mass_2'][1])
                log_J += lj

            if 'phase' in self._search_angles:
                a = self._search_angles['phase']
                x_prime[a['x']][inv] *= -1
                x_prime[a['y']][inv] *= -1

            if all(t in self._search_angles for t in ['tilt_1', 'tilt_2']):
                t1 = self._search_angles['tilt_1']
                t2 = self._search_angles['tilt_2']
                x_prime[[t1['x'], t1['y'], t2['x'], t2['y']]][inv] = \
                    x_prime[[t2['x'], t2['y'], t1['x'], t1['y']]][inv]

            elif any(t in self._search_angles for t in ['tilt_1', 'tilt_2']):
                raise RuntimeError(
                    'Cannot use q-inversion with only one tilt angle')

        if 'sky' in self._reparameterisations:
            x['ra'], x['dec'], r, lj = cartesian_to_sky(
                x_prime['sky_x'], x_prime['sky_y'], x_prime['sky_z'])
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

        if 'time' in self._reparameterisations:
            t, lj = inverse_rescale_minus_one_to_one(x_prime['time'],
                                                     self.time_bounds[0],
                                                     self.time_bounds[1])
            # This will break with casting rules
            x[self.time] = np.float64(t) + np.float64(self.time_offset)
            log_J += lj

        if self._search_angles:
            for a in self._search_angles.values():
                x[a['angle']], r, lj = cartesian_to_angle(
                    x_prime[a['x']], x_prime[a['y']], scale=a['scale'])
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

                if c['flip'] == 'lower':
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
                if f := c['flip']:
                    if f == 'both':
                        lower = x_prime[c['rescaled_name']] < 0.
                        upper = x_prime[c['rescaled_name']] > 1.
                        x[c['name']] = x_prime[c['rescaled_name']]
                        x[c['name']][lower] *= -1
                        x[c['name']][upper] = 2 - x[c['name']][upper]

                    else:
                        inv = x_prime[c['rescaled_name']] < 0.
                        x[c['name']][~inv] = x_prime[c['rescaled_name']][~inv]
                        x[c['name']][inv] = -x_prime[c['rescaled_name']][inv]

                        if c['flip'] == 'upper':
                            x[c['name']] = 1 - x[c['name']]

                    if c['rescale']:
                        x[c['name']], lj = \
                            inverse_rescale_zero_to_one(x[c['name']],
                                                        xmin=c['min'],
                                                        xmax=c['max'])
                        log_J += lj
                else:
                    if c['rescale']:
                        x[c['name']], lj = \
                            inverse_rescale_minus_one_to_one(x[c['name']],
                                                             xmin=c['min'],
                                                             xmax=c['max'])
                        log_J += lj
                    x[c['name']] = x_prime[c['rescaled_name']]

        if self.default_rescaling:
            for n in self.default_rescaling:
                # x[n], lj = \
                #     inverse_rescale_minus_one_to_one(x_prime[n + '_prime'],
                #                                      xmin=self._min[n],
                #                                      xmax=self._max[n])
                # log_J += lj
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
        return log_p


class AugmentedGWFlowProposal(GWFlowProposal):

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
