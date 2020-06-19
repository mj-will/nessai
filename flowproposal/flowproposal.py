
import logging
import numpy as np
from scipy.stats import chi

import torch

from .flowmodel import FlowModel


class FlowProposal:
    """
    Object that handles training and proposal points
    """
    def __init__(self, proposal_config=None, flow_config=None, output='./'):
        """
        Intialise
        """
        self.flow = FlowModel(config=flow_config, output=output)

        self.ndims=None
        self.populated = False
        self.fuzz = 1.0

        if latent_prior == 'gaussian':
            from .utils import draw_truncated_gaussian
            self.draw = draw_truncated_gassian
            self.log_prior = self._log_gaussian_prior
        elif latent_prior == 'uniform':
            from .utils import draw_random_nsphere
            self.draw = draw_random_nsphere
            self.log_prior = self._log_uniform_prior

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

    def train(self, live_points):
        """
        Train the normalising flow given the live points
        """
        # TODO: plots
        # TODO: weights resets
        data = self.rescale(live_points)
        self.flow.train(data)

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

    def radius2(self, z):
        """Calculate the radius of a latent_point"""
        return np.sum(z ** 2., axis=-1)

    def _log_uniform_prior(self, z):
        """
        Uniform prior for use with points drawn uniformly with an n-shpere
        """
        return 0.0

    def _log_gaussian_prior(self, z):
        """
        Gaussian prior
        """
        return np.sum(-0.5 * (z ** 2.) - 0.5 * log(2. * pi), axis=-1)

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
        log_p = self.log_prior(x)
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def populate(self, worst_point, N=10000):
        """Populate a pool of latent points"""
        worst_z, _ = self.forward_pass(worst_point)
        r2 = self.radius2(worst_z)
        logging.debug("Populating proposal")

        self.x= np.empty([0, self.ndims])
        self.z = np.empty([0, self.ndims])
        while len(self.samples) < N:
            while True:
                z = self.draw(old_r2, N, fuzz=self.fuzz)
                if z.size:
                    break

            x, log_J = self.backward_pass(z, rescale=True)
            # rescale given priors used intially, need for priors
            log_w = self.compute_weights(x, z, log_J)
            log_u = np.log(np.random.rand(N))
            indices = np.where((log_w - log_u) >= 0)[0]
            if not len(indices):
                logging.error('Rejection sampling produced zero samples!')
                raise RuntimeError('Rejection sampling produced zero samples!')
            if len(indices) / N < 0.01:
                logging.warning('Rejection sampling accepted less than 1 percent of samples!')
            else:
                # array of indices to take random draws from
                self.x += np.concatenate([self.x, x[indices]], axis=0)
                self.z = np.concatenate([self.z, z[indices]], axis=0)

        self.x = self.x[:N]
        self.z = self.z[:N]
        self.indices = np.random.permutation(N)
        self.populated = True

    def draw(self, worst_point):
        """
        Draw a replacement point
        """
        if not self.populated:
            while not self.populated
                self.populate(worst_point, N=self.proposal_size)
            self.count += 1

        # new sample is drawn randomly from proposed points
        index = self.indices.pop(0)
        new_sample = self.samples[index]
        if not self.indices.size:
            self.populated = False
            loggin.debug('Proposal pool is empty')
        # make live point and return
        return new_sample
