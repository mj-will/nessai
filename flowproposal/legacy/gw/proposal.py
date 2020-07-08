import logging
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.stats import chi

from ..proposal import CPNestFlowProposal
from ...utils import replace_in_list, rescale_zero_to_one, \
        inverse_rescale_zero_to_one, rescale_minus_one_to_one, \
        inverse_rescale_minus_one_to_one
from ...gw.utils import angle_to_cartesian, cartesian_to_angle, \
        sky_to_cartesian, cartesian_to_sky

logger = logging.getLogger(__name__)


class GWFlowProposal(CPNestFlowProposal):
    """
    A proposal specific to gravitational wave CBC
    """
    def __init__(self, model, reparameterisations={}, **kwargs):
        super(GWFlowProposal, self).__init__(model, **kwargs)

        self.set_reparameterisations(reparameterisations)
        # list to itnernally track reparemeterisations
        self._reparameterisations = []
        self._search_angles = {}

    def set_reparameterisations(self, reparameterisations):
        """
        Set the relevant reparamterisation flags
        """
        defaults = dict(mass_inversion=True, flip=False, reduced_quaternions=True,
                distance_rescaling=False, norm_quaternions=False, rescale_angles=True,
                euler_convention='ZYZ', angular_decomposition=True,
                minus_one_to_one=True)
        defaults.update(reparameterisations)
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

        self._search_angles[name] = {'angle': name, 'radial': radial_name,
                'x': x_name, 'y': y_name, 'scale': scale}

        logger.debug(f'Added {name} with config: {self._search_angles[name]}')

    def set_rescaling(self):
        """
        Set the rescaling functions
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()
        self.default_rescaling = []

        if self.angular_decomposition:
            if all(p in self.names for p in['ra', 'dec']):
                replace_in_list(self.rescaled_names, ['ra', 'dec'], ['sky_x', 'sky_y'])
                if 'luminosity_distance' not in self.names:
                    self.names.append('sky_radial')
                    self.distance = 'sky_radial'
                    self.rescaled_names.append('sky_z')
                else:
                    self.distance = 'luminosity_distance'
                    replace_in_list(self.rescaled_names, [self.distance],
                                ['sky_z'])
                self._reparameterisations.append('sky')

            elif any(p in self.names for p in ['ra', 'dec']):
                raise RuntimeError('Cannot use angular decompoisiton with only'
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
                    if (radial := f'a_{i}') in self.names:
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
            self.default_rescaling += list(set(self.names) & set(self.rescaled_names))
            replace_in_list(self.rescaled_names, self.default_rescaling,
                    [d + '_prime' for d in self.default_rescaling])

        self.rescale_parameters = 'all'
        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def rescale(self, x):
        """
        Rescale from the x space to the x prime space
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = np.zeros(x_prime.size)

        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']

        if self.default_rescaling:
            for n in self.default_rescaling:
                x_prime[n + '_prime'], lj = rescale_minus_one_to_one(x[n],
                        xmin=self.model.bounds[n][0],
                        xmax=self.model.bounds[n][1])
                log_J += lj

        if 'sky' in self._reparameterisations:
            if self.distance == 'luminosity_distance':
                r, lj = rescale_zero_to_one(x[self.distance],
                        xmin=self.model.bounds[self.distance][0],
                        xmax=self.model.bounds[self.distance][1])
                log_J += lj
            else:
                r = None

            x_prime['sky_x'], x_prime['sky_y'], x_prime['sky_z'], lj= \
                    sky_to_cartesian(x['ra'], x['dec'], r)
            log_J += lj

        if 'time' in self._reparameterisations:
            t = x[self.time] - self.time_offset
            x_prime['time'], lj = rescale_minus_one_to_one(t,
                    self.time_bounds[0], self.time_bounds[1])
            log_J += lj

        if self._search_angles:
            for a in self._search_angles.values():
                # if the radial parameter is present in x
                # use it, else samples with be drawn from a chi with
                # 2 d.o.f
                if (n := a['radial']) in self.model.names:
                    r, lj = rescale_zero_to_one(x[n],
                            self.model.bounds[n][0], self.model.bounds[n][1])
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
                x['mass_1'], lj = rescale_minus_one_to_one(x['mass_1'],
                        xmin=self.model.bounds['mass_1'][0],
                        xmax=self.model.bounds['mass_1'][1])
                log_J += lj
                x['mass_2'], lj = rescale_minus_one_to_one(x['mass_2'],
                        xmin=self.model.bounds['mass_2'][0],
                        xmax=self.model.bounds['mass_2'][1])
                log_J += lj
                x_prime_inv = x_prime.copy()
                x_prime[['mass_1_dbl', 'mass_2_dbl']] = x[['mass_1', 'mass_2']]
                x_prime_inv[['mass_1_dbl', 'mass_2_dbl']] = x[['mass_2', 'mass_1']]

            # flip the phase
            if 'phase' in self._search_angles:
                s = self._search_angles['phase']
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
                raise RuntimeError('Cannot use q-inversion with only one tilt angle')

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
                x['mass_ratio'][~inv] = np.exp(x_prime['mass_ratio_inv'][~inv])
                x['mass_ratio'][inv] = np.exp(-x_prime['mass_ratio_inv'][inv])
                # for q_inv < 0 conversion is exp(q_inv) for q_inv > 0 exp(-q_inv)
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
                raise RuntimeError('Cannot use q-inversion with only one tilt angle')

        if 'sky' in self._reparameterisations:
            x['ra'], x['dec'], r, lj = cartesian_to_sky(x_prime['sky_x'],
                    x_prime['sky_y'], x_prime['sky_z'])
            log_J += lj

            if self.distance == 'luminosity_distance':
                r, lj = inverse_rescale_zero_to_one(r,
                        xmin=self.model.bounds[self.distance][0],
                        xmax=self.model.bounds[self.distance][1])
                log_J += lj
            x[self.distance] = r

        if 'time' in self._reparameterisations:
            t, lj = inverse_rescale_minus_one_to_one(x_prime['time'],
                    self.time_bounds[0], self.time_bounds[1])
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
                    r, lj = inverse_rescale_zero_to_one(r,
                        self.model.bounds[n][0], self.model.bounds[n][1])
                    log_J += lj
                x[a['radial']] = r

        if self.default_rescaling:
            for n in self.default_rescaling:
                x[n], lj = inverse_rescale_minus_one_to_one(x_prime[n + '_prime'],
                        xmin=self.model.bounds[n][0],
                        xmax=self.model.bounds[n][1])
                log_J += lj

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
        return log_p

    def radius(self, z):
        """Calculate the radius of a latent_point"""
        return np.max(np.sqrt(np.sum(z ** 2., axis=-1)))

