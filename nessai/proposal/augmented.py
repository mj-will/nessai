import logging
import os

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from ..flowmodel import FlowModel
from ..livepoint import (
    live_points_to_array,
    )

from ..plot import plot_live_points, plot_acceptance
from .flowproposal import FlowProposal

logger = logging.getLogger(__name__)


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
        log_w -= np.nanmax(log_w)
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
            z_samples = np.empty([0, self.dims])
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
                    z_samples = np.empty([0, self.dims])
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
            z_samples = np.concatenate([z_samples, z[indices]], axis=0)
            if counter % 10 == 0:
                logger.debug(f'Accepted {self.x.size} / {N} points')
            counter += 1

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
        self.logger.info(
            f'Proposal populated with {len(self.indices)} samples')
