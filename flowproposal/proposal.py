import logging
import os

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from .flowmodel import FlowModel, update_config
from .livepoint import (
        live_points_to_array,
        numpy_array_to_live_points,
        get_dtype,
        DEFAULT_FLOAT_DTYPE
        )
from .plot import plot_live_points, plot_acceptance
from .utils import get_uniform_distribution, detect_edge

logger = logging.getLogger(__name__)


def _initialize_global_variables(model):
    """
    Store a global copy of the model for multiprocessing.
    """
    global _model
    _model = model


def _log_likelihood_wrapper(x):
    return _model.evaluate_log_likelihood(x)


class Proposal:

    def __init__(self, model):
        self.model = model
        self.populated = True
        self.initialised = False
        self.training_count = 0
        self.population_acceptance = None
        self.pool = None

    def initialise(self):
        """Initialise"""
        self.initialised = True

    def draw(self, old_param):
        return None

    def train(self, x, **kwargs):
        logger.info('This proposal method cannot be trained')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state


class AnalyticProposal(Proposal):

    def draw(self, old_param):
        """
        Draw directly from the analytic priors
        """
        return self.model.new_point()


class RejectionProposal(Proposal):

    def __init__(self, model, poolsize=1000):
        super(RejectionProposal, self).__init__(model)
        self.poolsize = poolsize
        self.populated = False
        self._acceptance_checked = True

    def draw_proposal(self):
        """Draw from the proposal distribution"""
        return self.model.new_point(N=self.poolsize)

    def log_proposal(self, x):
        """Proposal probability"""
        return self.model.log_prior(x)

    def get_weights(self, x):
        """Get weights for the samples"""
        log_p = self.model.log_prior(x)
        log_q = self.log_proposal(x)
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    @property
    def population_acceptance(self):
        if self._acceptance_checked:
            return None
        else:
            self._acceptance_checked = True
            return self._population_acceptance

    @population_acceptance.setter
    def population_acceptance(self, acceptance):
        self._population_acceptance = acceptance
        self._acceptance_checked = False

    def populate(self):
        """Populate"""
        x = self.draw_proposal()
        log_w = self.get_weights(x)
        log_u = np.log(np.random.rand(self.poolsize))
        indices = np.where((log_w - log_u) >= 0)[0]
        self.samples = x[indices]
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        self.population_acceptance = self.samples.size / self.poolsize
        self.populated = True

    def draw(self, old_sample):
        """Propose a new sample"""
        if not self.populated:
            self.populate()
        # get new sample
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
        return new_sample


class FlowProposal(RejectionProposal):
    """
    Object that handles training and proposal points
    """

    def __init__(self, model, flow_config=None, output='./', poolsize=10000,
                 rescale_parameters=False, latent_prior='truncated_gaussian',
                 fuzz=1.0, keep_samples=False, exact_poolsize=True, plot=True,
                 fixed_radius=False, drawsize=10000, check_acceptance=False,
                 truncate=False, zero_reset=None, detect_edges=False,
                 rescale_bounds=[-1, 1], rescale_min_max=False,
                 boundary_inversion=False, inversion_type='duplicate',
                 update_bounds=True, max_radius=False, pool=None, n_pool=None,
                 multiprocessing=False, **kwargs):
        """
        Initialise
        """
        super(FlowProposal, self).__init__(model)
        logger.debug('Initialising FlowProposal')

        self.flow = None
        self._flow_config = None
        self.initialised = False
        self.populated = False
        self.indices = []
        self.training_count = 0
        self.populated_count = 0
        self.names = []
        self.x = None
        self.z = None
        self.samples = None
        self.rescaled_names = []

        self.output = output
        self.poolsize = poolsize
        self.drawsize = drawsize
        self.fuzz = fuzz
        self.latent_prior = latent_prior
        self.rescale_parameters = rescale_parameters
        self.keep_samples = keep_samples
        self.exact_poolsize = exact_poolsize
        self.rescale_min_max = rescale_min_max
        self.update_bounds = update_bounds
        self.check_acceptance = check_acceptance
        self.rescale_bounds = rescale_bounds
        self.truncate = truncate
        self.zero_reset = zero_reset
        self.boundary_inversion = boundary_inversion
        self.inversion_type = inversion_type
        self.detect_edges = detect_edges
        self._edges = {}

        self.pool = pool
        self.n_pool = n_pool
        if multiprocessing:
            if not self.check_acceptance:
                self.check_acceptance = True
            self._setup_pool()

        if self.detect_edges:
            self._edge_mode_range = 0.1
            self._edge_cutoff = 0.1
        else:
            # Will always return an edge
            self._edge_mode_range = 0.0
            self._edge_cutoff = 0.0

        if plot:
            if isinstance(plot, str):
                if plot == 'all':
                    self._plot_pool = True
                    self._plot_training = True
                elif plot == 'train':
                    self._plot_pool = False
                    self._plot_training = True
                elif plot == 'pool':
                    self._plot_pool = True
                    self._plot_training = False
                else:
                    logger.warning(
                        f'Unknown plot argument: {plot}, setting all false'
                        )
                    self._plot_pool = False
                    self._plot_training = False
            else:
                self._plot_pool = True
                self._plot_training = True

        else:
            self._plot_pool = False
            self._plot_training = False

        self.acceptance = []
        self.approx_acceptance = []
        self.flow_config = flow_config

        if self.latent_prior == 'truncated_gaussian':
            from .utils import draw_truncated_gaussian
            self.draw_latent_prior = draw_truncated_gaussian
        elif self.latent_prior == 'gaussian':
            logger.warning('Using a gaussian latent prior WITHOUT truncation')
            from .utils import draw_gaussian
            self.draw_latent_prior = draw_gaussian
        elif self.latent_prior == 'uniform':
            from .utils import draw_uniform
            self.draw_latent_prior = draw_uniform
        elif self.latent_prior == 'uniform_nsphere':
            from .utils import draw_nsphere
            self.draw_latent_prior = draw_nsphere
        else:
            raise RuntimeError(
                f'Unknown latent prior: {self.latent_prior}, choose from: '
                'truncated_gaussian (default), gaussian, '
                'uniform, uniform_nsphere'
                )
        # Alternative latent distribution for use with uniform sphere
        # Allows for training with Gaussian prior and sampling with
        # uniform prior
        self.alt_dist = None

        if fixed_radius:
            try:
                self.fixed_radius = float(fixed_radius)
            except ValueError:
                logger.error(
                    'Fixed radius enabled but could not be converted to a '
                    'float. Setting fixed_radius=False'
                    )
                self.fixed_radius = False
        else:
            self.fixed_radius = False

        self.max_radius = max_radius

    def _setup_pool(self):
        """
        Setup the multiprocessing pool for
        """
        if self.pool is None:
            import multiprocessing
            self.pool = multiprocessing.Pool(
                processes=self.n_pool,
                initializer=_initialize_global_variables,
                initargs=(self.model,)
            )

    def _close_pool(self):
        if getattr(self, "pool", None) is not None:
            logger.info("Starting to close worker pool.")
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Finished closing worker pool.")

    @property
    def flow_config(self):
        """Return the configuration for the flow"""
        return self._flow_config

    @flow_config.setter
    def flow_config(self, config):
        """Set configuration (includes checking defaults)"""
        self._flow_config = update_config(config)

    @property
    def dims(self):
        """Return the number of dimensions"""
        return len(self.names)

    @property
    def rescaled_dims(self):
        """Return the number of rescaled dimensions"""
        return len(self.rescaled_names)

    @property
    def x_dtype(self):
        """Return the dtype for the x space"""
        return get_dtype(self.names, DEFAULT_FLOAT_DTYPE)

    @property
    def x_prime_dtype(self):
        """Return the dtype for the x prime space"""
        return get_dtype(self.rescaled_names, DEFAULT_FLOAT_DTYPE)

    def initialise(self):
        """
        Initialise the proposal class
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.set_rescaling()
        self.verify_rescaling()
        self.flow_config['model_config']['n_inputs'] = self.rescaled_dims
        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()
        # if rescale, update names

        if (b := self.boundary_inversion):
            if not self.rescale_parameters:
                raise RuntimeError('Boundary inversion requires rescaling')

            if not isinstance(b, list):
                if isinstance(self.rescale_parameters, list):
                    b = self.rescale_parameters
                else:
                    b = self.names.copy()
            else:
                if not set(b).issubset(self.names):
                    raise RuntimeError(
                            'Boundaries are not in known parameters')
            self.boundary_inversion = b
            logger.info(
                    'Appyling boundary inversion to: '
                    f'{self.boundary_inversion}'
                    )

            if self.inversion_type not in ('split', 'duplicate'):
                raise RuntimeError(
                        f'Unknown inversion type: {self.inversion_type}')

            self.rescale_bounds = [0, 1]
            self.update_bounds = True
            self._edges = {n: None for n in b}
            logger.info(f'Changing bounds to {self.rescale_bounds}')
        else:
            self.boundary_inversion = []

        if self.rescale_parameters:
            # if rescale is a list, there are the parameters to rescale
            # else all parameters are rescale
            if not isinstance(self.rescale_parameters, list):
                self.rescale_parameters = self.names.copy()
            for i, rn in enumerate(self.rescaled_names):
                if rn in self.rescale_parameters:
                    self.rescaled_names[i] += '_prime'

            self._min = {n: self.model.bounds[n][0] for n in self.model.names}
            self._max = {n: self.model.bounds[n][1] for n in self.model.names}
            self._rescale_factor = np.ptp(self.rescale_bounds)
            self._rescale_shift = self.rescale_bounds[0]

            self.rescale = self._rescale_to_bounds
            self.inverse_rescale = self._inverse_rescale_to_bounds
            logger.info(f'Set to rescale inputs to {self.rescale_bounds}')

            if self.update_bounds:
                logger.info(
                        'Rescaling will use min and max of current live points'
                        )
            else:
                logger.info('Rescaling will use model bounds')

        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def verify_rescaling(self):
        """
        Verify the rescaling functions
        """
        logger.info('Verifying rescaling functions')
        x = self.model.new_point(N=5000)
        x_prime, log_J = self.rescale(x)
        x_out, log_J_inv = self.inverse_rescale(x_prime)

        if x.size == x_out.size:
            for f in x.dtype.names:
                if not np.allclose(x[f], x_out[f]):
                    raise RuntimeError(f'Rescaling is not invertible for {f}')
            if not np.allclose(log_J, -log_J_inv):
                raise RuntimeError('Rescaling Jacobian is not invertible')
        else:
            ratio = x_out.size // x.size
            for f in x.dtype.names:
                if not any([np.allclose(x_out[f][:x.size],
                                        x_out[f][n * x.size:(n + 1) * x.size])
                            for n in range(1, ratio)]):
                    raise RuntimeError(
                        'Duplicate samples to map to same input values. '
                        'Check the rescaling and inverse rescaling functions.')
            for f in x.dtype.names:
                if not np.allclose(x[f], x_out[f][:x.size]):
                    raise RuntimeError(f'Rescaling is not invertible for {f}')
            if not np.allclose(log_J, -log_J_inv):
                raise RuntimeError('Rescaling Jacobian is not invertible')

        logger.info('Rescaling functions are invertible')

    def _rescale_to_bounds(self, x):
        """
        Rescale the inputs to specified bounds
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = np.zeros(x_prime.size)

        if x.size == 1:
            x = np.array([x], dtype=x.dtype)

        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x_prime[rn] = self._rescale_factor \
                             * ((x[n] - self._min[n])
                                / (self._max[n] - self._min[n])) \
                             + self._rescale_shift

                log_J += (-np.log(self._max[n] - self._min[n])
                          + np.log(self._rescale_factor))

                if n in self.boundary_inversion:

                    if self._edges[n] is None:
                        self._edges[n] = detect_edge(
                            x_prime[rn],
                            bounds=[0, 1],
                            cutoff=self._edge_cutoff,
                            mode_range=self._edge_mode_range
                            )

                    if self._edges[n]:
                        logger.debug(
                            f'Apply inversion for {n} to '
                            f'{self._edges[n]} bound'
                            )
                        if self._edges[n] == 'upper':
                            x_prime[rn] = 1 - x_prime[rn]
                        if self.inversion_type == 'duplicate':
                            x_inv = x_prime.copy()
                            x_inv[rn] *= -1
                            x_prime = np.concatenate([x_prime, x_inv])
                            x = np.concatenate([x,  x])
                        else:
                            inv = np.random.choice(x_prime.size,
                                                   x_prime.size // 2,
                                                   replace=False)
                            x_prime[rn][inv] *= -1
                    else:
                        logger.debug(f'Not using inversion for {n}')
            else:
                x_prime[rn] = x[n]
        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']
        return x_prime, log_J

    def _inverse_rescale_to_bounds(self, x_prime):
        """
        Rescale the inputs from the prime space to the phyiscal space
        using the bounds specified
        """
        x = np.zeros([x_prime.size], dtype=self.x_dtype)
        log_J = np.zeros(x_prime.size)
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                if n in self.boundary_inversion:
                    inv = x_prime[rn] < 0.
                    x_prime[rn][~inv] = x_prime[rn][~inv]
                    x_prime[rn][inv] = -x_prime[rn][inv]

                    if self._edges[n] == 'upper':
                        x_prime[rn] = 1 - x_prime[rn]

                x[n] = (self._max[n] - self._min[n]) \
                    * (x_prime[rn] - self._rescale_shift) \
                    / self._rescale_factor + self._min[n]

                log_J += (np.log(self._max[n] - self._min[n])
                          - np.log(self._rescale_factor))
            else:
                x[n] = x_prime[rn]
        x['logP'] = x_prime['logP']
        x['logL'] = x_prime['logL']
        return x, log_J

    def rescale(self, x):
        """
        Rescale from the phyisical space to the primed physical
        space
        """
        log_J = np.zeros(x.size)
        return x, log_J

    def inverse_rescale(self, x_prime):
        """
        Rescale from the primed phyisical space to the original physical
        space
        """
        log_J = np.zeros(x_prime.size)
        return x_prime, log_J

    def check_state(self, x):
        """
        Skeleton function for operations that need to checked before training
        """
        if self.update_bounds:
            self._min = {n: np.min(x[n]) for n in self.names}
            self._max = {n: np.max(x[n]) for n in self.names}
        if self.boundary_inversion:
            self._edges = {n: None for n in self.boundary_inversion}

    def train(self, x, plot=True):
        """
        Train the normalising flow given the live points
        """

        block_output = self.output + f'/training/block_{self.training_count}/'
        if not os.path.exists(block_output):
            os.makedirs(block_output, exist_ok=True)

        self.check_state(x)

        if self._plot_training:
            plot_live_points(x, c='logL',
                             filename=block_output + 'x_samples.png')

        x_prime, log_J = self.rescale(x)

        if self.rescale_parameters and self._plot_training:
            plot_live_points(x_prime, c='logL',
                             filename=block_output + 'x_prime_samples.png')
        # Convert to numpy array for training and remove likelihoods and priors
        # Since the names of parameters may have changes, pull names from flows
        x_prime = live_points_to_array(x_prime, self.rescaled_names)
        self.flow.train(x_prime, output=block_output, plot=self._plot_training)

        if self._plot_training:
            self.alt_dist = None
            z, _ = self.flow.sample_and_log_prob(N=5000)
            x_prime, log_prob = self.backward_pass(z, rescale=False)
            x_prime['logL'] = log_prob
            plot_live_points(
                x_prime,
                c='logL',
                filename=block_output + 'x_prime_generated.png'
                )
            x, log_J = self.inverse_rescale(x_prime)
            x, log_J = self.check_prior_bounds(x, log_J)
            x['logL'] += log_J
            if self.rescale_parameters:
                plot_live_points(x, c='logL',
                                 filename=block_output + 'x_generated.png')

        self.populated = False
        self.training_count += 1

    def reset_model_weights(self):
        """
        Reset the flows weights
        """
        self.flow.reset_model()

    def check_prior_bounds(self, x, *args):
        """
        Return only values that are within the prior bounds
        """
        idx = np.array(list(((x[n] >= self.model.bounds[n][0])
                             & (x[n] <= self.model.bounds[n][1]))
                       for n in self.model.names)).T.all(1)
        out = (a[idx] for a in (x,) + args)
        return out

    def forward_pass(self, x, rescale=True):
        """Pass a vector of points through the model"""
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x)
            log_J += log_J_rescale
        x = live_points_to_array(x, names=self.rescaled_names)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        z, log_prob = self.flow.forward_and_log_prob(x)
        return z, log_prob + log_J

    def backward_pass(self, z, rescale=True):
        """A backwards pass from the model (latent -> real)"""
        # Compute the log probability
        x, log_prob = self.flow.sample_and_log_prob(z=z,
                                                    alt_dist=self.alt_dist)
        x = numpy_array_to_live_points(x.astype(DEFAULT_FLOAT_DTYPE),
                                       self.rescaled_names)
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_prob -= log_J
            x, log_prob = self.check_prior_bounds(x, log_prob)
        return x, log_prob

    def radius(self, z, log_q=None):
        """Calculate the radius of a latent_point"""
        if log_q is not None:
            r = np.sqrt(np.sum(z ** 2., axis=-1))
            i = np.argmax(r)
            return r[i], log_q[i]
        else:
            return np.mean(np.sqrt(np.sum(z ** 2., axis=-1)))

    def log_prior(self, x):
        """
        Compute the prior probability
        """
        return self.model.log_prior(x)

    def compute_weights(self, x, log_q):
        """
        Compute the weight for a given set of samples
        """
        log_p = self.log_prior(x)
        x['logP'] = log_p
        x['logL'] = log_q
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def populate(self, worst_point, N=10000, plot=True):
        """Populate a pool of latent points"""
        if self.fixed_radius:
            r = self.fixed_radius
        else:
            worst_z, worst_q = self.forward_pass(worst_point, rescale=True)
            r, worst_q = self.radius(worst_z, worst_q)
            if self.max_radius:
                if r > self.max_radius:
                    r = self.max_radius
            logger.debug(f'Populating proposal with lantent radius: {r:.5}')

        if self.latent_prior == 'uniform_nsphere':
            self.alt_dist = get_uniform_distribution(self.dims, r)

        warn = True
        warn_zero = True
        if not self.keep_samples or not self.indices:
            self.x = np.array([], dtype=self.x_dtype)
            self.z = np.empty([0, self.dims])
        counter = 0
        zero_counter = 0
        proposed = 0
        while len(self.x) < N:
            while True:
                z = self.draw_latent_prior(self.dims, r=r, N=self.drawsize,
                                           fuzz=self.fuzz)
                if z.size:
                    break
            proposed += z.shape[0]
            x, log_q = self.backward_pass(z, rescale=True)
            if self.truncate:
                cut = log_q >= worst_q
                x = x[cut]
                log_q = log_q[cut]
            # rescale given priors used initially, need for priors
            log_w = self.compute_weights(x, log_q)
            log_u = np.log(np.random.rand(x.shape[0]))
            indices = np.where((log_w - log_u) >= 0)[0]

            if not len(indices) or (len(indices) / self.drawsize < 0.001):
                if warn_zero:
                    logger.warning(
                        'Rejection sampling produced almost zero samples!'
                        )
                    warn_zero = False
                zero_counter += 1
                if (zero_counter == self.zero_reset and
                        self.x.size < (N // 2) and
                        self.latent_prior == 'truncated_gaussian'):
                    logger.warning(
                        'Proposal is too ineffcient, reducing radius'
                        )
                    r *= 0.99
                    logger.warning(f'New radius: {r}')
                    self.x = np.array([], dtype=self.x_dtype)
                    self.z = np.empty([0, self.dims])
                    zero_counter = 0
                    continue
            if len(indices) / self.drawsize < 0.01:
                if warn:
                    logger.warning(
                        'Rejection sampling accepted less than 1 percent of '
                        f'samples! ({len(indices) / self.drawsize})')
                    warn = False

            # array of indices to take random draws from
            self.x = np.concatenate([self.x, x[indices]], axis=0)
            self.z = np.concatenate([self.z, z[indices]], axis=0)
            if counter % 10 == 0:
                logger.debug(f'Accepted {self.x.size} / {N} points')
            counter += 1

        if self.exact_poolsize:
            self.x = self.x[:N]
            self.z = self.z[:N]

        if self._plot_pool:
            plot_live_points(
                self.x, c='logL',
                filename=f'{self.output}/pool_{self.populated_count}.png'
                )

        self.samples = self.x[self.model.names + ['logP', 'logL']]

        if self.check_acceptance:
            self.approx_acceptance.append(self.compute_acceptance(worst_q))
            logger.debug(
                f'Current approximate acceptance {self.approx_acceptance[-1]}'
                )
            self.evaluate_likelihoods()
            self.acceptance.append(
                self.compute_acceptance(worst_point['logL'])
                )
            logger.debug(f'Current acceptance {self.acceptance[-1]}')
            if self._plot_pool:
                plot_acceptance(
                    self.approx_acceptance,
                    self.acceptance,
                    labels=['approx', 'analytic'],
                    filename=f'{self.output}/proposal_acceptance.png'
                    )
        else:
            self.samples['logL'] = np.zeros(self.samples.size,
                                            dtype=self.samples['logL'].dtype)

        self.indices = np.random.permutation(self.samples.size).tolist()
        self.population_acceptance = self.x.size / proposed
        self.populated_count += 1
        self.populated = True
        logger.debug(f'Proposal populated with {len(self.indices)} samples')
        logger.debug(
                f'Overall proposal acceptance: {self.x.size / proposed:.4}'
                )

    def evaluate_likelihoods(self):
        """
        Evaluate the likelihoods for the pool of live points
        """
        if self.pool is None:
            for s in self.samples:
                s['logL'] = self.model.evaluate_log_likelihood(s)
        else:
            self.samples['logL'] = self.pool.map(_log_likelihood_wrapper,
                                                 self.samples)
            self.model.likelihood_evaluations += self.samples.size

    def compute_acceptance(self, logL):
        """
        Compute how many of the current pool have log-likelihoods greater
        than the specified value
        """
        return (self.samples['logL'] > logL).sum() / self.samples.size

    def draw(self, worst_point):
        """
        Draw a replacement point
        """
        if not self.populated:
            while not self.populated:
                self.populate(worst_point, N=self.poolsize)
        # new sample is drawn randomly from proposed points
        # popping from right end is faster
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
            logger.debug('Proposal pool is empty')
        # make live point and return
        return new_sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state['initialised'] = False
        state['weights_file'] = state['flow'].weights_file
        # Mask may be generate via permutation, so must be saved
        if 'mask' in state['flow'].model_config['kwargs']:
            state['mask'] = state['flow'].model_config['kwargs']['mask']
        else:
            state['mask'] = None
        # user provides model and config for resume
        # flow can be reconstructed from resume
        del state['pool']
        del state['model']
        del state['_flow_config']
        del state['flow']
        for a in ['x', 'z', 'indices', 'samples']:
            if a in state:
                del state[a]
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class AugmentedFlowProposal(FlowProposal):

    def __init__(self, model, augment_features=1, **kwargs):
        super().__init__(model, **kwargs)
        self.augment_features = augment_features

    def set_rescaling(self):
        super().set_rescaling()
        self.augment_names = [f'e_{i}' for i in range(self.augment_features)]
        self.names += self.augment_names
        self.rescaled_names += self.augment_names
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

    def _rescale_with_bounds(self, x, generate_augment=True):
        """
        Rescale the inputs to [-1, 1] using the bounds as the min and max
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = 0.
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x_prime[rn] = 2 * ((x[n] - self.model.bounds[n][0])
                                   / (self.model.bounds[n][1]
                                      - self.model.bounds[n][0])) - 1

                log_J += np.log(2) - np.log(self.model.bounds[n][1]
                                            - self.model.bounds[n][0])
            elif n not in self.augment_names:
                x_prime[rn] = x[n]
        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']
        if generate_augment:
            if x_prime.size == 1:
                x_prime[self.augment_names] = self.augment_features * (0.,)
            else:
                for an in self.augment_names:
                    x_prime[an] = np.random.randn(x_prime.size)

        return x_prime, log_J

    def _inverse_rescale_with_bounds(self, x_prime):
        """
        Rescale the inputs from the prime space to the phyiscal space
        using the model bounds
        """
        x = np.zeros([x_prime.size], dtype=self.x_dtype)
        log_J = 0.
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x[n] = (self.model.bounds[n][1] - self.model.bounds[n][0]) \
                        * ((x_prime[rn] + 1) / 2) + self.model.bounds[n][0]
                log_J += np.log(self.model.bounds[n][1]
                                - self.model.bounds[n][0]) - np.log(2)
            else:
                x[n] = x_prime[rn]
        x['logP'] = x_prime['logP']
        x['logL'] = x_prime['logL']
        return x, log_J

    def augmented_prior(self, x):
        """
        Log guassian for augmented variables
        """
        logP = 0.
        for n in self.augment_names:
            logP += stats.norm.logpdf(x[n])
        return logP

    def _marginalise_augment(self, x, K=50):
        """
        Marginalise out the augmented feautures
        """
        x_prime, log_J = self.rescale(x, generate_augment=False)
        x_prime = live_points_to_array(x_prime, names=self.rescaled_names)
        x_prime = np.repeat(x_prime, K, axis=0)

        x_prime[:, -self.augment_features:] = np.random.randn(
                x.size * K, self.augment_features)
        z, log_prob = self.flow.forward_and_log_prob(x_prime)
        log_prob = log_prob.reshape(K, x.size)
        log_prob_e = np.sum(stats.norm.logpdf(
            x_prime[:, -self.augment_features:]), axis=-1
            ).reshape(K, x.size)
        assert log_prob.shape == log_prob_e.shape
        log_prob = -np.log(K) + logsumexp(log_prob - log_prob_e, (0))
        return log_prob - log_J

    def log_prior(self, x):
        """
        Compute the prior probability
        """
        physical_prior = self.model.log_prior(x[self.model.names])
        augmented_prior = self.augmented_prior(x)
        # return physical_prior
        return physical_prior + augmented_prior

    def compute_weights(self, x, log_q):
        """
        Compute the weight for a given set of samples
        """
        # log_q = self._marginalise_augment(x)
        log_p = self.log_prior(x)
        x['logP'] = log_p
        x['logL'] = log_q
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w, log_q

    def populate(self, worst_point, N=10000, plot=True):
        """Populate a pool of latent points"""
        if self.fixed_radius:
            r = self.fixed_radius
        else:
            worst_z, worst_q = self.forward_pass(worst_point, rescale=True)
            # worst_log_q = self._marginalise_augment(worst_point)
            # worst_log_q = worst_q
            r = self.radius(worst_z)
        logger.debug(f'Populating proposal with lantent radius: {r:.5}')
        warn = True
        if not self.keep_samples or not self.indices:
            self.x = np.array([], dtype=self.x_dtype)
            self.z = np.empty([0, self.dims])
        counter = 0
        zero_counter = 0
        while len(self.x) < N:
            while True:
                z = self.draw_latent_prior(self.dims, r=r, N=self.drawsize,
                                           fuzz=self.fuzz)
                if z.size:
                    break

            x, log_q = self.backward_pass(z, rescale=True)
            # rescale given priors used initially, need for priors
            log_w, log_q = self.compute_weights(x, log_q)
            # x = x[log_q > worst_log_q]
            # log_w = log_w[log_q > worst_log_q]
            log_u = np.log(np.random.rand(x.shape[0]))
            indices = np.where((log_w - log_u) >= 0)[0]

            if not len(indices) or (len(indices) / self.drawsize < 0.001):
                logger.warning(
                    'Rejection sampling produced almost zero samples!')
                zero_counter += 1
                if zero_counter == 10 and self.x.size < (N // 2) and \
                        self.latent_prior != 'uniform':
                    logger.warning(
                        'Proposal is too ineffcient, ''reducing radius')
                    r *= 0.99
                    logger.warning(f'New radius: {r}')
                    self.x = np.array([], dtype=self.x_dtype)
                    self.z = np.empty([0, self.dims])
                    zero_counter = 0
                    continue
            if len(indices) / self.drawsize < 0.01:
                if warn:
                    logger.warning(
                        'Rejection sampling accepted less than 1 percent of '
                        f'samples! ({len(indices) / self.drawsize})'
                        )
                    warn = False

            # array of indices to take random draws from
            self.x = np.concatenate([self.x, x[indices]], axis=0)
            self.z = np.concatenate([self.z, z[indices]], axis=0)
            if counter % 10 == 0:
                logger.debug(f'Accepted {self.x.size} / {N} points')
            counter += 1

        if self.exact_poolsize:
            self.x = self.x[:N]
            self.z = self.z[:N]

        if plot:
            plot_live_points(
                self.x,
                filename=f'{self.output}/pool_{self.populated_count}.png'
                )

        self.samples = self.x[self.model.names + ['logP', 'logL']]

        if self.check_acceptance:
            self.approx_acceptance.append(self.compute_acceptance(worst_q))
            logger.debug('Current approximate acceptance: '
                         f'{self.approx_acceptance[-1]}')
            self.evaluate_likelihoods()
            self.acceptance.append(
                self.compute_acceptance(worst_point['logL']))
            logger.debug(f'Current acceptance {self.acceptance[-1]}')
            if plot:
                plot_acceptance(
                    self.approx_acceptance, self.acceptance,
                    labels=['approx', 'analytic'],
                    filename=f'{self.output}/proposal_acceptance.png'
                    )
        else:
            self.samples['logL'] = np.zeros(self.samples.size,
                                            dtype=self.samples['logL'].dtype)

        self.indices = np.random.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        logger.debug(f'Proposal populated with {len(self.indices)} samples')
