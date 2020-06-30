import os
import logging
import numpy as np
import torch

from .flowmodel import FlowModel, update_config
from .livepoint import live_points_to_array, numpy_array_to_live_points, get_dtype
from .plot import plot_live_points

logger = logging.getLogger(__name__)

class Proposal:

    def __init__(self, model):
        self.model = model
        self.populated = True
        self.intialised = False

    def initialise(self):
        """Initialise"""
        self.intialised = True

    def draw(self, old_param):
        return None

    def train(self, x, **kwargs):
        logger.info('This proposal method cannot be trained')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['initialised'] = False
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
        self.poolsize=poolsize
        self.populated = False

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

    def populate(self):
        """Populate"""
        x = self.draw_proposal()
        log_w = self.get_weights(x)
        log_u = np.log(np.random.rand(self.poolsize))
        indices = np.where((log_w - log_u) >= 0)[0]
        self.samples = x[indices]
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
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


class FlowProposal(Proposal):
    """
    Object that handles training and proposal points
    """

    def __init__(self, model, flow_config=None, output='./', poolsize=10000,
            rescale_parameters=False, latent_prior='gaussian', fuzz=1.0,
            keep_samples=False, exact_poolsize=True, plot=True, **kwargs):
        """
        Initialise
        """
        super(FlowProposal, self).__init__(model)
        logger.debug('Initialising FlowProposal')

        self.flow = None
        self.initialised = False
        self.populated = False
        self.indices = []
        self.training_count = 0
        self.populated_count = 0
        self.names = []
        self.x = None
        self.z = None
        self.rescaled_names = []

        self.output = output
        self.poolsize = poolsize
        self.fuzz = fuzz
        self.latent_prior = latent_prior
        self.rescale_parameters = rescale_parameters
        self.keep_samples = keep_samples
        self.exact_poolsize = exact_poolsize

        self.flow_config = update_config(flow_config)

        if self.latent_prior == 'gaussian':
            from .utils import draw_truncated_gaussian
            self.draw_latent_prior = draw_truncated_gaussian
            self.log_latent_prior = self._log_gaussian_prior
        elif self.latent_prior == 'uniform':
            from .utils import draw_random_nsphere
            self.draw_latent_prior = draw_random_nsphere
            self.log_latent_prior = self._log_uniform_prior

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
        return get_dtype(self.names, 'f8')

    @property
    def x_prime_dtype(self):
        """Return the dtype for the x prime space"""
        return get_dtype(self.rescaled_names, 'f8')

    def initialise(self):
        """
        Initialise the proposal class
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
        self.set_rescaling()
        self.flow_config['model_config']['n_inputs'] = self.rescaled_dims
        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.initialised = True

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()
        # if rescale, update names
        if self.rescale_parameters:
            # if rescale is a list, there are the parameters to rescale
            # else all parameters are rescale
            if not isinstance(self.rescale_parameters, list):
                self.rescale_parameters = self.names
            for i, rn in enumerate(self.rescaled_names):
                if rn in self.rescale_parameters:
                    self.rescaled_names[i] += '_prime'
            self.rescale = self._rescale_with_bounds
            self.inverse_rescale = self._inverse_rescale_with_bounds

        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

    def _rescale_with_bounds(self, x):
        """
        Rescale the inputs to [-1, 1] using the bounds as the min and max
        """
        x_prime = np.zeros([x.size], dtype=self.x_prime_dtype)
        log_J = 0.
        for n, rn in zip(self.names, self.rescaled_names):
            if n in self.rescale_parameters:
                x_prime[rn] = 2 * ((x[n] - self.model.bounds[n][0]) \
                    / (self.model.bounds[n][1] - self.model.bounds[n][0])) - 1

                log_J += np.log(2) - np.log(self.model.bounds[n][1] \
                        - self.model.bounds[n][0])
            else:
                x_prime[rn] = x[n]
        x_prime['logP'] = x['logP']
        x_prime['logL'] = x['logL']
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
                log_J += np.log(self.model.bounds[n][1] - self.model.bounds[n][0]) \
                        - np.log(2)
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
        log_J = 0.
        return x, log_J

    def inverse_rescale(self, x_prime):
        """
        Rescale from the primed phyisical space to the original physical
        space
        """
        log_J = 0.
        return x_prime, log_J

    def train(self, x, plot=True):
        """
        Train the normalising flow given the live points
        """
        block_output = self.output + f'/training/block_{self.training_count}/'
        if not os.path.exists(block_output):
            os.makedirs(block_output, exist_ok=True)

        if plot:
            plot_live_points(x, filename=block_output + 'x_samples.png')

        x_prime, log_J = self.rescale(x)

        if self.rescale_parameters and plot:
            plot_live_points(x_prime, filename=block_output + 'x_prime_samples.png')
        # Convert to numpy array for training and remove likelihoods and priors
        # Since the names of parameters may have changes, pull names from flows
        x_prime = live_points_to_array(x_prime, self.rescaled_names)
        self.flow.train(x_prime, output=block_output, plot=plot)

        if plot:
            z = np.random.randn(5000, self.rescaled_dims)
            x_prime, log_J = self.backward_pass(z, rescale=False)
            plot_live_points(x_prime, filename=block_output + 'x_prime_generated.png')
            x, log_J = self.inverse_rescale(x_prime)
            plot_live_points(x, filename=block_output + 'x_generated.png')

        self.populated = False
        self.training_count += 1

    def reset_model_weights(self):
        """
        Reset the flows weights
        """
        self.flow.reset_model()

    def forward_pass(self, x, rescale=True):
        """Pass a vector of points through the model"""
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x)
            log_J += log_J_rescale
        x = live_points_to_array(x, names=self.rescaled_names)
        x_tensor = torch.Tensor(x.astype(np.float32)).to(self.flow.device)
        self.flow.model.eval()
        with torch.no_grad():
            z, log_J_tensor = self.flow.model(x_tensor, mode='direct')
        z = z.detach().cpu().numpy().astype('f8')
        log_J += log_J_tensor.detach().cpu().numpy().astype('f8')
        return z, np.squeeze(log_J)

    def backward_pass(self, z, rescale=True):
        """A backwards pass from the model (latent -> real)"""
        z_tensor = torch.Tensor(z.astype(np.float32)).to(self.flow.device)
        self.flow.model.eval()
        with torch.no_grad():
            theta, log_J = self.flow.model(z_tensor, mode='inverse')
        x = theta.detach().cpu().numpy().astype('f8')
        log_J = log_J.detach().cpu().numpy().astype('f8')
        x = numpy_array_to_live_points(x, self.rescaled_names)
        if rescale:
            x, log_J_rescale = self.inverse_rescale(x)
            log_J += log_J_rescale
        return x, np.squeeze(log_J)

    def radius(self, z):
        """Calculate the radius of a latent_point"""
        return np.sqrt(np.sum(z ** 2., axis=-1))

    def _log_uniform_prior(self, z):
        """
        Uniform prior for use with points drawn uniformly with an n-shpere
        """
        return 0.0

    def _log_gaussian_prior(self, z):
        """
        Gaussian prior
        """
        return np.sum(-0.5 * (z ** 2.) - 0.5 * np.log(2. * np.pi), axis=-1)

    def log_proposal_prob(self, z, log_J):
        """
        Compute the proposal probaility for a given point

        """
        # Since the Jacobian is for Z -> X, we use the inverse
        log_q_z = self.log_latent_prior(z)
        return log_q_z - log_J

    def log_prior(self, x):
        """
        Compute the prior probability
        """
        return self.model.log_prior(x)

    def compute_weights(self, x, z, log_J):
        """
        Compute the weight for a given set of samples
        """
        log_q = self.log_proposal_prob(z, log_J)
        log_p = self.log_prior(x)
        x['logP'] = log_p
        x['logL'] = log_q
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def populate(self, worst_point, N=10000, plot=True):
        """Populate a pool of latent points"""
        worst_z, _ = self.forward_pass(worst_point, rescale=True)
        r = self.radius(worst_z)
        logger.debug("Populating proposal")
        if not self.keep_samples or not self.indices:
            self.x = np.array([], dtype=self.x_dtype)
            self.z = np.empty([0, self.dims])
        while len(self.x) < N:
            while True:
                z = self.draw_latent_prior(self.dims, r, N, fuzz=self.fuzz)
                if z.size:
                    break

            x, log_J = self.backward_pass(z, rescale=True)
            # rescale given priors used intially, need for priors
            log_w = self.compute_weights(x, z, log_J)
            log_u = np.log(np.random.rand(x.shape[0]))
            indices = np.where((log_w - log_u) >= 0)[0]
            if not len(indices):
                logger.error('Rejection sampling produced zero samples!')
                raise RuntimeError('Rejection sampling produced zero samples!')
            if len(indices) / N < 0.01:
                logger.warning('Rejection sampling accepted less than 1 percent of samples!')
            else:
                # array of indices to take random draws from
                self.x = np.concatenate([self.x, x[indices]], axis=0)
                self.z = np.concatenate([self.z, z[indices]], axis=0)

        if self.exact_poolsize:
            self.x = self.x[:N]
            self.z = self.z[:N]

        if plot:
            plot_live_points(self.x,
                    filename=f'{self.output}/pool_{self.populated_count}.png')

        self.indices = np.random.permutation(self.x.size).tolist()
        self.populated = True
        logger.debug(f'Proposal populated with {len(self.indices)} samples')

    def evaluate_likelihoods(self):
        """
        Evaluate the likelihoods for the pool of live points
        """
        pass

    def draw(self, worst_point):
        """
        Draw a replacement point
        """
        if not self.populated:
            while not self.populated:
                self.populate(worst_point, N=self.poolsize)
            self.populated_count += 1
        # new sample is drawn randomly from proposed points
        # popping from right end is faster
        index = self.indices.pop()
        new_sample = self.x[index]
        if not self.indices:
            self.populated = False
            logger.debug('Proposal pool is empty')
        # make live point and return
        return new_sample
