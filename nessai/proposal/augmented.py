# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from ..flowmodel import FlowModel
from ..livepoint import (
    live_points_to_array,
    )

from .flowproposal import FlowProposal

logger = logging.getLogger(__name__)


class AugmentedFlowProposal(FlowProposal):

    def __init__(self, model, augment_features=1, generate_augment='gaussian',
                 **kwargs):
        super().__init__(model, **kwargs)
        self.augment_features = augment_features
        self.generate_augment = generate_augment
        self.marginalise_augment = False

    def set_rescaling(self):
        super().set_rescaling()
        self._base_rescale = self.rescale
        self.rescale = self._augmented_rescale
        self.augment_names = [f'e_{i}' for i in range(self.augment_features)]
        self.names += self.augment_names
        self.rescaled_names += self.augment_names
        logger.info(f'augmented x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(
            f'Augmented x prime space parameters: {self.rescaled_names}')

    def initialise(self):
        """
        Initialise the proposal class
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self._x_dtype = False
        self._x_prime_dtype = False

        self.set_rescaling()
        self.verify_rescaling()
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info('Overwritting fuzz factor with expansion fraction')
            self.fuzz = \
                (1 + self.expansion_fraction) ** (1 / self.rescaled_dims)
            logger.info(f'New fuzz factor: {self.fuzz}')

        m = np.ones(self.rescaled_dims)
        m[-self.augment_features:] = -1
        self.flow_config['model_config']['kwargs']['mask'] = m

        self.flow_config['model_config']['n_inputs'] = self.rescaled_dims
        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def _augmented_rescale(self, x, generate_augment=None,
                           compute_radius=False, **kwargs):
        """Rescale with augment parameter.

        Parameters
        ----------
        x : array
            Structured array in X space with augment parameters in the
            fields.
        generate_augment : {None, 'gaussian', 'zeros', 'zeroes'}, optional
            Method used to generate the augmented parameters. If None Gaussian
            is used if compute_radius=False else attribute of the same name
            is used.
        compute_radius : bool, optional
            Boolean to indicate when rescale is being used for computing a
            radius.
        """
        x_prime, log_J = self._base_rescale(x, compute_radius=compute_radius,
                                            **kwargs)

        if generate_augment is None:
            if compute_radius:
                generate_augment = self.generate_augment
            else:
                generate_augment = 'gaussian'

        if generate_augment in ['zeros', 'zeroes']:
            for an in self.augment_names:
                x_prime[an] = np.zeros(x_prime.size)
        elif generate_augment == 'gaussian':
            for an in self.augment_names:
                x_prime[an] = np.random.randn(x_prime.size)
        return x_prime, log_J

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
            x_prime[:, -self.augment_features:]), axis=-1).reshape(K, x.size)
        assert log_prob.shape == log_prob_e.shape
        log_prob = -np.log(K) + logsumexp(log_prob - log_prob_e, (0))
        return log_prob - log_J

    def augmented_prior(self, x):
        """
        Log guassian for augmented variables
        """
        logP = 0.
        for n in self.augment_names:
            logP += stats.norm.logpdf(x[n])
        return logP

    def log_prior(self, x):
        """
        Compute the prior probability
        """
        return (self.model.log_prior(x[self.model.names]) +
                self.augmented_prior(x))

    def rejection_sampling(self, z, worst_q=None):
        """
        Perform rejection sampling.

        Coverts samples from the latent space and computes the corresponding
        weights. Then returns samples using standard rejection sampling.

        Parameters
        ----------
        z :  ndarray
            Samples from the latent space
        worst_q : float, optional
            Lower bound on the log-probability computed using the flow that
            is used to truncate new samples. Not recommended.

        Returns
        -------
        array_like
            Array of accepted latent samples.
        array_like
            Array of accepted samples in the X space.
        """
        if self.marginalise_augment:
            raise NotImplementedError
        x, log_q = self.backward_pass(z, rescale=not self.use_x_prime_prior)

        if not x.size:
            return x, log_q

        if self.truncate:
            cut = log_q >= worst_q
            x = x[cut]
            log_q = log_q[cut]

        # rescale given priors used initially, need for priors
        log_w = self.compute_weights(x, log_q)
        log_u = np.log(np.random.rand(x.shape[0]))
        indices = np.where(log_w >= log_u)[0]

        return z[indices], x[indices]
