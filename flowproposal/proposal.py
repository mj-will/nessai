import os
import logging
import numpy as np
import torch

from .flowmodel import FlowModel, update_config
from .livepoint import live_points_to_array, numpy_array_to_live_points

logger = logging.getLogger(__name__)

class Proposal:

    def __init__(self, model):
        self.model = model
        self.populated = True

    def initialise(self):
        """Initialise"""
        pass

    def draw(self, old_param):
        return None

    def train(self, x):
        logger.info('This proposal method cannot be trained')


class AnalyticProposal(Proposal):

    def draw(self, old_param):
        """
        Draw directly from the analytic priors
        """
        return self.model.sample()


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
            latent_prior='gaussian', fuzz=1.0, **kwargs):
        """
        Initialise
        """
        super(FlowProposal, self).__init__(model)
        logger.debug('Initialising FlowProposal')

        self.flow = None
        self.initialised = False
        self.populated = False
        self.training_count = 0
        self.populated_count = 0

        self.output = output
        self.dims = model.dims
        self.poolsize = poolsize
        self.fuzz = fuzz
        self.latent_prior = latent_prior

        flow_config = update_config(flow_config)
        flow_config['model_config']['n_inputs'] = self.dims
        self.flow = FlowModel(config=flow_config, output=output)

        if self.latent_prior == 'gaussian':
            from .utils import draw_truncated_gaussian
            self.draw_latent_prior = draw_truncated_gaussian
            self.log_latent_prior = self._log_gaussian_prior
        elif self.latent_prior == 'uniform':
            from .utils import draw_random_nsphere
            self.draw_latent_prior = draw_random_nsphere
            self.log_latent_prior = self._log_uniform_prior

    def initialise(self):
        """
        Initialise the proposal class
        """
        self.flow.initialise()
        self.initialised = True

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

    def train(self, x):
        """
        Train the normalising flow given the live points
        """
        # TODO: plots
        # TODO: weights resets
        block_output = self.output + f'/block_{self.training_count}/'
        data, log_J = self.rescale(x)
        self.flow.train(data, output=block_output)
        self.training_count += 1

    def forward_pass(self, x, rescale=True):
        """Pass a vector of points through the model"""
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x)
            log_J += log_J_rescale
        x_tensor = torch.Tensor(x.astype(np.float32)).to(self.flow.device)
        with torch.no_grad():
            z, log_J_tensor = self.flow.model(x_tensor, mode='direct')
        z = z.detach().cpu().numpy()
        log_J += log_J_tensor.detach().cpu().numpy()
        return z, np.squeeze(log_J)

    def backward_pass(self, z, rescale=True):
        """A backwards pass from the model (latent -> real)"""
        z_tensor = torch.Tensor(z.astype(np.float32)).to(self.flow.device)
        with torch.no_grad():
            theta, log_J = self.flow.model(z_tensor, mode='inverse')
        x = theta.detach().cpu().numpy()
        log_J = log_J.detach().cpu().numpy()
        if rescale:
            x, log_J_rescale = self.inverse_rescale(x)
            # TODO: add jacobian here
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
        Compute the proposal probaility for a given point assuming the latent
        distribution is a unit gaussian

        q(theta)  = q(z)|dz/dtheta|
        """
        log_q_z = self.log_latent_prior(z)
        return log_q_z + log_J

    def compute_weights(self, x, z, log_J):
        """
        Compute the weight for a given set of samples
        """
        log_q = self.log_proposal_prob(z, log_J)
        log_p = self.model.log_prior(x)
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def populate(self, worst_point, N=10000):
        """Populate a pool of latent points"""
        worst_point = live_points_to_array(worst_point, self.model.names)
        worst_z, _ = self.forward_pass(worst_point, rescale=True)
        r = self.radius(worst_z)
        logger.debug("Populating proposal")

        self.x = np.array([], dtype=[(n, 'f') for n in self.model.names + ['logL', 'logP']])
        self.z = np.empty([0, self.dims])
        while len(self.x) < N:
            while True:
                z = self.draw_latent_prior(self.dims, r, N, fuzz=self.fuzz)
                if z.size:
                    break

            x, log_J = self.backward_pass(z, rescale=True)
            x = numpy_array_to_live_points(x, names=self.model.names)
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

        self.x = self.x[:N]
        self.z = self.z[:N]
        self.indices = np.random.permutation(N).tolist()
        self.populated = True

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
