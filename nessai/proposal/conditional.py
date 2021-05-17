# -*- coding: utf-8 -*-
"""Conditional version of FlowProposal"""
import logging

import numpy as np

from .flowproposal import FlowProposal
from ..livepoint import live_points_to_array
from ..utils import (
    InterpolatedDistribution,
    rescale_zero_to_one,
    )


logger = logging.getLogger(__name__)


class ConditionalFlowProposal(FlowProposal):
    """Conditional version of FlowProposal.

    In nessai parameter which are not included directly in the mapping are
    refered to as conditional parameters. However, in ``nflows`` the term
    context is used instead. As such, ``FlowModel`` uses both terms since
    it interfaces with ``nflows``.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-define model
    conditional_likelihood : bool, optional
        If True the likelihood is included as a conditional input to the flow.
    kwargs :
        Keyword arguments passed to :obj:`~nessai.proposal.FlowProposal`
    """
    def __init__(self, model, conditional_likelihood=False, **kwargs):
        super(ConditionalFlowProposal, self).__init__(model, **kwargs)

        self.conditional_parameters = []
        self.conditional_likelihood = conditional_likelihood
        self.conditional = self.conditional_likelihood

        self._min_logL = None
        self._max_logL = None

    @property
    def conditional_dims(self):
        return len(self.conditional_parameters)

    @property
    def rescaled_worst_logL(self):
        if self.worst_logL is not None:
            return ((self.worst_logL - self._min_logL) /
                    (self._max_logL - self._min_logL))
        else:
            return None

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        super().set_rescaling()
        self.configure_likelihood_parameter()

    def check_state(self, x):
        """
        Operations that need to checked before training. These include
        updating the bounds for rescaling and resetting the bounds for
        inversion.

        Parameters
        ----------
        x: array_like
            Array of training live points which can be used to set parameters
        """
        super().check_state(x)
        if self.update_bounds or self._min_logL is None:
            self._min_logL = np.min(x['logL'])
            self._max_logL = np.max(x['logL'])

    def configure_likelihood_parameter(self):
        """Configure the likelihood parameter"""
        if self.conditional_likelihood:
            self.likelihood_index = len(self.conditional_parameters)
            self.conditional_parameters += ['logL']
            self.likelihood_distribution = \
                InterpolatedDistribution('logL', rescale=False)

    def update_flow_config(self):
        """Update the flow configuration dictionary.

        Calls the parent method first and then adds the conditional dimensions.
        """
        super().update_flow_config()
        if self.conditional:
            self.flow_config['model_config']['kwargs']['context_features'] = \
                    self.conditional_dims

    def reset_reparameterisation(self):
        """Reset the model.

        This resets the stored min and max logL and calls the parent method.
        """
        super().reset_reparameterisation()
        self._min_logL = None
        self._max_logL = None

    def train_on_data(self, x_prime, output):
        """
        Function that takes live points converts to numpy array and calls
        the train function. Live points should be in the X' (x prime) space.
        """
        x_prime_array = live_points_to_array(x_prime, self.rescaled_names)
        conditional = self.get_conditional(x_prime)
        self.train_conditional(conditional)
        self.flow.train(x_prime_array, conditional=conditional, output=output,
                        plot=self._plot_training)

    def train_conditional(self, conditional):
        """Update the methods for sampling the conditional"""
        if self.conditional_likelihood:
            self.likelihood_distribution.update_samples(
                    conditional[:, self.likelihood_index],
                    reset=self.update_bounds)

    def sample_conditional_parameters(self, n):
        """
        Draw n samples from the conditional distributions.
        """
        conditional = np.empty([n, self.conditional_dims])
        if self.conditional_likelihood:
            conditional[:, self.likelihood_index] = \
                self.likelihood_distribution.sample(
                    n, min_logL=self.rescaled_worst_logL)

        return conditional

    def get_conditional(self, x):
        """
        Get the conditional parameters if empty return None


        Includes likelihood rescaling to [0, 1].
        """
        if not self.conditional:
            return
        conditional = np.empty([x.size, self.conditional_dims])
        if self.conditional_likelihood:
            conditional[:, self.likelihood_index] = rescale_zero_to_one(
                x['logL'].flatten(), self._min_logL, self._max_logL)[0]

        if conditional.size:
            return conditional

    def forward_pass(self, x, conditional=None, **kwargs):
        """
        Pass a vector of points through the flow model.

        Calls the parent method with a conditional. Context is either specified
        or retrived using `get_conditional`

        Parameters
        ----------
        x : array_like
            Live points to map to the latent space
        conditional : array_like, optional
            Context array passed to the flow.
        kwargs :
            Keyword arguments passed to the parent method

        Returns
        -------
        x : array_like
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            jacobian)
        """
        if conditional is None and self.conditional:
            conditional = self.get_conditional(x)
        z, log_prob = super().forward_pass(
            x, conditional=conditional, **kwargs)
        return z, log_prob

    def backward_pass(self, z, conditional=None, **kwargs):
        """
        A backwards pass from the model (latent -> real)

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space
        conditional : array_like, optional
            Context array passed to the flow.
        kwargs :
            Keyword arguments passed to the parent method

        Returns
        -------
        x : array_like
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            Jacobian)
        """
        if conditional is None and self.conditional:
            conditional = self.sample_conditional_parameters(z.shape[0])

        x, log_prob = super().backward_pass(
            z, conditional=conditional, **kwargs)
        return x, log_prob
