import logging
import numpy as np
from scipy.stats import chi

from ..proposal import FlowProposal
from ..utils import replace_in_list, rescale_zero_to_one, \
        inverse_rescale_zero_to_one, rescale_minus_one_to_one, \
        inverse_rescale_minus_one_to_one
from .utils import angle_to_cartesian, cartesian_to_angle, \
        sky_to_cartesian, cartesian_to_sky

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

    def set_reparameterisations(self, reparameterisations):
        """
        Set the relevant reparamterisation flags
        """
        defaults = dict(q_inversion=True, flip=False, reduced_quaternions=True,
                distance_rescaling=False, norm_quaternions=False, rescale_angles=True,
                euler_convention='ZYZ', angular_decomposition=True)
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

        if self.angular_decomposition:
            if all(p in self.names for p in['ra', 'dec']):
                replace_in_list(self.rescaled_names, ['ra', 'dec'], ['sky_x', 'sky_y'])
                if 'luminosity_distance' not in self.names:
                    self.names.append('sky_radial')
                    self.distance = 'sky_radial'
                    self.rescaled_names.append('sky_z')
                else:
                    self.distance = 'luminosity_distance'
                    replace_in_list(self.rescaaled_names, [self.distance],
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

        self.rescale_parameters = 'all'
        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def rescale(self, x):
        """
        Rescale from the x space to the x prime space
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = 0.

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

        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']

        return x_prime, log_J

    def inverse_rescale(self, x_prime):
        """
        Rescale from the x prime  space to the x space
        """
        x = np.zeros([x_prime.size], dtype=self.x_dtype)
        log_J = 0.

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
                    r, lj = inverse_resacle_zero_to_one(r,
                        self.model.bounds[n][0], self.model.bounds[n][0])
                    log_J += lj
                x[a['radial']] = r

        x['logP'] = x_prime['logP']
        x['logL'] = x_prime['logL']

        return x, log_J

    def log_prior(self, x):
        """
        Modified log prior that handles radial parameters
        """
        log_p = self.model.log_prior(x[self.model.names])

        if 'sky' in self._reparameterisations:
            if self.distance == 'sky_radial':
                log_p += chi.logpdf(x[self.distance], 3)
        #if self.angle_indices:
        #    for i in self.angle_indices:
        #        if not i[2]:
        #            log_p + chi.logpdf(radial_components[:, i[1]-self.base_dim], 2)
        return log_p

