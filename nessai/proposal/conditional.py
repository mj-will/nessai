# -*- coding: utf-8 -*-
"""Conditional version of FlowProposal"""
import logging
import numpy as np

from .flowproposal import FlowProposal

from ..distributions import (
    InterpolatedDistribution,
    CategoricalDistribution
)
from ..livepoint import live_points_to_array
from ..utils import rescale_zero_to_one


logger = logging.getLogger(__name__)


class ConditionalFlowProposal(FlowProposal):
    """Conditional version of FlowProposal.

    In nessai parameters which are not included directly in the mapping are
    refered to as conditional parameters. However, in ``nflows`` the term
    context is used instead. As such, ``FlowModel`` uses both terms since
    it interfaces with ``nflows``.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-define model
    conditional_likelihood : bool, optional
        If True the likelihood is included as a conditional input to the flow.
    categorical_parameters : list, optional
        List of parameters in the model which are to be treated as
        cateogorical, i.e. they can onlt take a small set of fixed value.
        Alternatively, this can be set directly in the model.
    kwargs :
        Keyword arguments passed to :obj:`~nessai.proposal.FlowProposal`
    """
    def __init__(
        self,
        model,
        conditional_likelihood=False,
        categorical_parameters=None,
        **kwargs
    ):

        super(ConditionalFlowProposal, self).__init__(model, **kwargs)

        self.conditional_parameters = []
        self.conditional_likelihood = conditional_likelihood
        self.conditional = False
        self.categorical_parameters = categorical_parameters

        self._min_logL = None
        self._max_logL = None
        self._flow_names = None

    @property
    def conditional_dims(self):
        """Number of conditional parameters."""
        return len(self.conditional_parameters)

    @property
    def flow_names(self):
        """Names of parameters use in the flow sampling space.

        This excludes the conditional parameters
        """
        return self._flow_names

    @flow_names.setter
    def flow_names(self, names):
        if not names == self.rescaled_names[:len(names)]:
            raise RuntimeError(
                'Flow names must be the first n rescaled names. '
                f'Received: {names}, rescaled names: {self.rescaled_names}'
            )
        else:
            self._flow_names = names

    @property
    def flow_dims(self):
        """Dimensions of the flow."""
        return len(self._flow_names)

    @property
    def rescaled_worst_logL(self):
        """Rescaled version of the current worst log-likelihood value"""
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
        self.configure_categorical_parameters()
        self.flow_names = [n for n in self.rescaled_names
                           if n not in self.conditional_parameters]
        # Make sure the parameters are in the correct order
        if not (self.flow_names + self.categorical_parameters ==
                self.rescaled_names):
            raise RuntimeError('Parameters do not match')

        if not self.conditional:
            raise RuntimeError('No conditional parameters in the proposal!')

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
            self.conditional = True

    def configure_categorical_parameters(self):
        """Configure the categorical parameters"""
        if self.model.categorical_parameters is not None:
            logger.debug('Using categorical parameters from model')
            self.categorical_parameters = self.model.categorical_parameters

        if self.categorical_parameters is not None:
            logger.debug(
                f'Adding categorial parameters: {self.categorical_parameters}'
            )
            n = len(self.categorical_parameters)
            self.categorical_distribution = CategoricalDistribution()
            self.categorical_indices = \
                (np.arange(n) + len(self.conditional_parameters)).tolist()
            self.conditional_parameters += self.categorical_parameters
            self.conditional = True
        else:
            logger.debug('No categorial parameters')
            self.categorical_parameters = []

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
        x_prime_array = live_points_to_array(x_prime, self.flow_names)
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
        if self.categorical_parameters:
            self.categorical_distribution.update_samples(
                conditional[:, self.categorical_indices],
                reset=self.update_bounds
            )

    def sample_conditional_parameters(self, n):
        """Draw n samples from the conditional distributions.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        """
        conditional = np.empty([n, self.conditional_dims])
        log_prob = np.zeros(n)
        if self.conditional_likelihood:
            conditional[:, self.likelihood_index] = \
                self.likelihood_distribution.sample(
                    n, min_logL=self.rescaled_worst_logL)

        if self.categorical_parameters:
            conditional[:, self.categorical_indices], lp = \
                self.categorical_distribution.sample(n)
            log_prob += lp

        return conditional, log_prob

    def get_conditional(self, x):
        """
        Get the conditional parameters if empty return None


        Includes likelihood rescaling to [0, 1].
        """
        if not self.conditional or not self.conditional_dims:
            return
        conditional = np.empty([x.size, self.conditional_dims])
        if self.conditional_likelihood:
            conditional[:, self.likelihood_index] = rescale_zero_to_one(
                x['logL'].flatten(), self._min_logL, self._max_logL)[0]

        if self.categorical_parameters:
            logger.debug('Getting categorical parameters')
            conditional[:, self.categorical_indices] = \
                live_points_to_array(x, names=self.categorical_parameters)

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

    def _backward_pass(self, z, conditional, **kwargs):
        """Adapted backward pass of the flow that includes condtional \
            parameters.

        Specifically this method adds the conditional parameters that are
        also defined in the model back to the output vector.
        """
        try:
            x, log_prob = self.flow.sample_and_log_prob(
                z=z, alt_dist=self.alt_dist, conditional=conditional, **kwargs)
            x = np.concatenate([x, conditional[:, self.categorical_indices]],
                               axis=1)
            return x, log_prob
        except AssertionError:
            return np.array([]), np.array([])

    def backward_pass(self, z, conditional=None, log_prob=None, **kwargs):
        """
        A backwards pass from the model (latent -> real)

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space
        conditional : array_like, optional
            Context array passed to the flow.
        log_prob : :obj:`numpy.ndarray`
            Array of probabilities for the input samples. Useful when
            conditional samples are included in the model and should therefore
            be included in the flow probability.
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
        if log_prob is None:
            log_prob = np.zeros(z.shape[0])
        if conditional is None and self.conditional:
            conditional, conditional_log_prob = \
                self.sample_conditional_parameters(z.shape[0])
            log_prob += conditional_log_prob
        x, log_prob = super().backward_pass(
            z,
            conditional=conditional,
            log_prob=log_prob,
            **kwargs
        )
        return x, log_prob
