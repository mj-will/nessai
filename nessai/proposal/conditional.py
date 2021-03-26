# -*- coding: utf-8 -*-
"""Conditional version of FlowProposal"""
import logging
import os

import numpy as np

from .flowproposal import FlowProposal

from ..flowmodel import FlowModel
from ..livepoint import (
    live_points_to_array,
    numpy_array_to_live_points,
    DEFAULT_FLOAT_DTYPE
    )
from ..utils import (
    InterpolatedDistribution,
    rescale_minus_one_to_one
    )


logger = logging.getLogger(__name__)


class ConditionalFlowProposal(FlowProposal):
    """Conditional version of FlowProposal.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-define model
    conditional_likelihood : bool, optional
        If True the likelihood is included as a conditional input to the flow.
    kwargs :
        Keyword arguments passed to :obj:`~nessai.proposal.FlowProposal`
    """
    def __init__(self, model, uniform_parameters=False,
                 conditional_likelihood=False, transform_likelihood=False,
                 **kwargs):
        super(ConditionalFlowProposal, self).__init__(model, **kwargs)

        self.conditional_parameters = []

        if not uniform_parameters or uniform_parameters is None:
            self.uniform_parameters = []
        else:
            self.uniform_parameters = uniform_parameters

        self.conditional_likelihood = conditional_likelihood
        self.transform_likelihood = transform_likelihood

        self.conditional = any([self.uniform_parameters,
                                self.conditional_likelihood])

    @property
    def uniform_dims(self):
        """Number of uniform parameters"""
        return len(self.uniform_parameters)

    @property
    def flow_dims(self):
        """Return the number of dimensions for the normalising flow"""
        return self.rescaled_dims - self.uniform_dims

    @property
    def flow_names(self):
        return list(set(self.rescaled_names) - set(self.uniform_parameters))

    @property
    def conditional_dims(self):
        return len(self.conditional_parameters)

    def set_rescaling(self):
        """
        Set function and parameter names for rescaling
        """
        self.names = self.model.names.copy()
        self.rescaled_names = self.names.copy()

        self.set_uniform_parameters()
        self.set_likelihood_parameter()

        self.set_boundary_inversion()

        if self.rescale_parameters:
            # if rescale is a list, there are the parameters to rescale
            # else all parameters are rescale
            if not isinstance(self.rescale_parameters, list):
                self.rescale_parameters = self.rescaled_names.copy()

            self.rescale_parameters = list(set(self.rescale_parameters)
                                           - set(self.uniform_parameters))

            for i, rn in enumerate(self.rescaled_names):
                if rn in self.rescale_parameters:
                    self.rescaled_names[i] += '_prime'

            self._min = {n: self.model.bounds[n][0] for n in self.model.names}
            self._max = {n: self.model.bounds[n][1] for n in self.model.names}
            self._rescale_factor = np.ptp(self.rescale_bounds)
            self._rescale_shift = self.rescale_bounds[0]

            # self.rescale = self._rescale_to_bounds
            self.inverse_rescale = self._inverse_rescale_to_bounds
            logger.info(f'Set to rescale inputs to {self.rescale_bounds}')

            if self.update_bounds:
                logger.info(
                    'Rescaling will use min and max of current live points')
            else:
                logger.info('Rescaling will use model bounds')
        else:
            self.rescale_parameters = []

        logger.info(f'x space parameters: {self.names}')
        logger.info(f'parameters to rescale {self.rescale_parameters}')
        logger.info(f'x prime space parameters: {self.rescaled_names}')

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
        if self.update_bounds:
            self._min = {n: np.min(x[n]) for n in self.model.names}
            self._max = {n: np.max(x[n]) for n in self.model.names}
        if self.boundary_inversion:
            self._edges = {n: None for n in self.boundary_inversion}
        if self.transform_likelihood:
            self._min_logl = np.min(x['logL'])
            self._max_logl = np.max(x['logL'])

    def set_likelihood_parameter(self):
        if self.conditional_likelihood:
            self._use_logL = True
            self.likelihood_index = len(self.conditional_parameters)
            self.conditional_parameters += ['logL']
            self.likelihood_distribution = \
                InterpolatedDistribution('logL', rescale=True)
        if self.transform_likelihood:
            self.names.append('logL_train')
            self.rescaled_names.append('logL_train')

    def set_uniform_parameters(self):
        """
        Set the uniform parameters
        """
        if self.uniform_parameters:
            self.uniform_indices = np.arange(len(self.uniform_parametrs)) \
                                   + len(self.conditional_parameters)
            self.conditional_parameters += self.uniform_parameters
            self.uniform_min = \
                [self.model.bounds[n][0] for n in self.uniform_parameters]
            self.uniform_max = \
                [self.model.bounds[n][1] for n in self.uniform_parameters]

            self._uniform_log_prob = \
                -np.log(np.ptp([self.uniform_min, self.uniform_max]))
        else:
            logger.info('No uniform parameters to set')

    def initialise(self):
        """
        Initialise the proposal class.

        This includes:
            * Setting up the rescaling
            * Verifying the rescaling is invertible
            * Intitialising the FlowModel
        """
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.set_rescaling()
        self.verify_rescaling()
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info('Overwritting fuzz factor with expansion fraction')
            self.fuzz = \
                (1 + self.expansion_fraction) ** (1 / self.flow_dims)
            logger.info(f'New fuzz factor: {self.fuzz}')
        self.flow_config['model_config']['n_inputs'] = self.flow_dims
        if self.conditional:
            self.flow_config['model_config']['kwargs']['context_features'] = \
                    self.conditional_dims
        self.flow = FlowModel(config=self.flow_config, output=self.output)
        self.flow.initialise()
        self.populated = False
        self.initialised = True

    def add_auxiliary_parameters(self, x):
        """
        Add additional parameters which are not included in the default
        model.
        """
        if 'logL_train' in x.dtype.names:
            x['logL_train'], _ = rescale_minus_one_to_one(
                x['logL'], self._min_logl, self._max_logl)
        return x

    def rescale(self, x, compute_radius=False, test=None):
        x_prime, log_J = self._rescale_to_bounds(x, compute_radius, test=test)
        x_prime = self.add_auxiliary_parameters(x_prime)
        return x_prime, log_J

    def train_on_data(self, x_prime, output):
        """
        Function that takes live points converts to numpy array and calls
        the train function. Live points should be in the X' (x prime) space.
        """
        x_prime = self.add_auxiliary_parameters(x_prime)
        x_prime_array = live_points_to_array(x_prime, self.flow_names)
        context = self.get_context(x_prime)
        self.train_context(context)
        self.flow.train(x_prime_array, context=context, output=output,
                        plot=self._plot_training)

    def train_context(self, context):
        if self.conditional_likelihood:
            self.likelihood_distribution.update_samples(
                    context[:, self.likelihood_index],
                    reset=self.update_bounds)

    def sample_context_parameters(self, n):
        """
        Draw n samples from the context distributions
        """
        context = np.empty([n, self.conditional_dims])
        log_prob = np.zeros(n,)
        if self.uniform_parameters:
            u, log_prob_u = self.samples_uniform_parameters
            context[:, self.uniform_indices] = u
            log_prob += log_prob_u
        if self.conditional_likelihood:
            context[:, self.likelihood_index] = \
                self.likelihood_distribution.sample(
                    n, min_logL=self.worst_logL)

        return context, log_prob

    def sample_uniform_parameters(self, n):
        """
        Draw n parameters from a uniform distribution
        """
        if self.uniform_parameters:
            x = np.random.uniform(self.uniform_min, self.uniform_max,
                                  (n, self.uniform_dims))
            log_prob = self._uniform_log_prob * np.ones(n)
            return x, log_prob
        else:
            return None

    def get_context(self, x):
        """
        Get the context parameters if empty return None
        """
        context = np.empty([x.size, self.conditional_dims])
        if self.uniform_parameters:
            context[:, self.uniform_indices] = \
                live_points_to_array(x, self.uniform_parameters)
        if self.conditional_likelihood:
            context[:, self.likelihood_index] = x['logL'].flatten()

        if context.size:
            return context

    def forward_pass(self, x, rescale=True, compute_radius=True):
        """
        Pass a vector of points through the model

        Parameters
        ----------
        x : array_like
            Live points to map to the latent space
        rescale : bool, optional (True)
            Apply rescaling function
        compute_radius : bool, optional (True)
            Flag parsed to rescaling for rescaling specific to radius
            computation

        Returns
        -------
        x : array_like
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            jacobian)
        """
        log_J = 0
        if rescale:
            x, log_J_rescale = self.rescale(x, compute_radius=compute_radius)
            log_J += log_J_rescale

        x_flow = live_points_to_array(x, names=self.flow_names)
        context = self.get_context(x)

        if x_flow.ndim == 1:
            x_flow = x_flow[np.newaxis, :]
        if x_flow.shape[0] == 1:
            if self.clip:
                x_flow = np.clip(x_flow, *self.clip)
        z, log_prob = self.flow.forward_and_log_prob(x_flow, context=context)
        return z, log_prob + log_J

    def backward_pass(self, z, context=None, rescale=True, log_prob=0):
        """
        A backwards pass from the model (latent -> real)

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space
        rescale : bool, optional (True)
            Apply inverse rescaling function

        Returns
        -------
        x : array_like
            Samples in the latent sapce
        log_prob : array_like
            Log probabilties corresponding to each sample (including the
            Jacobian)
        """
        if context is None and self.conditional:
            context, log_prob_context = \
                self.sample_context_parameters(z.shape[0])
            log_prob += log_prob_context

        try:
            x_flow, log_prob_flow = self.flow.sample_and_log_prob(
                z=z, context=context, alt_dist=self.alt_dist)
        except AssertionError:
            return np.array([]), np.array([])

        log_prob += log_prob_flow
        if context is not None and self.uniform_parameters:
            x = np.concatenate([x_flow, context[:-1]], axis=1)
        else:
            x = x_flow

        valid = np.isfinite(log_prob)
        x, log_prob = x[valid], log_prob[valid]
        x = numpy_array_to_live_points(
            x.astype(DEFAULT_FLOAT_DTYPE), self.rescaled_names)
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_prob -= log_J
            x, log_prob = self.check_prior_bounds(x, log_prob)
        return x, log_prob
