# -*- coding: utf-8 -*-
"""
Proposals specifically for use with the importance based nested sampler.
"""
import copy
import logging
import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy

from nessai.plot import plot_histogram, plot_live_points

from .base import Proposal
from ..flowmodel import FlowModel, update_config
from ..livepoint import (
    get_dtype,
    live_points_to_array,
    numpy_array_to_live_points
)
from ..model import Model
from ..utils.rescaling import (
    gaussian_cdf_with_log_j,
    inv_gaussian_cdf_with_log_j,
    logit,
    sigmoid,
)
from ..utils.structures import get_subset_arrays, isfinite_struct


logger = logging.getLogger(__name__)


class ImportanceFlowProposal(Proposal):
    """Flow-based proposal for importance-based nested sampling.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-defined model.
    """
    def __init__(
        self,
        model: Model,
        output: str,
        initial_draws: int,
        use_logit: bool = False,
        reparam: str = None,
        plot_training: bool = False,
        weighted_kl: bool = True,
        weights_include_likelihood: bool = False,
        reset_flows: bool = False,
        flow_config: dict = None,
        combined_proposal: bool = True,
    ) -> None:
        self.flows = []
        self.level_count = -1
        self.draw_count = 0

        self.model = model
        self.output = output
        self.flow_config = flow_config
        self.plot_training = plot_training
        self.use_logit = use_logit
        self.reset_flows = reset_flows
        self.reparam = reparam
        self.weighted_kl = weighted_kl

        self.initial_draws = initial_draws
        self.initial_log_g = np.log(self.initial_draws)
        self.n_draws = {'initial': initial_draws}
        self.levels = {'initial': None}

        logger.debug(f'Initial g: {np.exp(self.initial_log_g)}')
        self._history = dict(poolsize=[], entropy=[])
        self.log_likelihood = None

        self.combined_proposal = combined_proposal
        self.weights_include_likelihood = weights_include_likelihood

        self.dtype = get_dtype(self.model.names)

    @property
    def flow_config(self) -> dict:
        """Return the configuration for the flow"""
        return self._flow_config

    @flow_config.setter
    def flow_config(self, config: dict) -> None:
        """Set configuration (includes checking defaults)"""
        if config is None:
            config = dict(model_config=dict())
        config['model_config']['n_inputs'] = self.model.dims
        self._flow_config = update_config(config)

    @property
    def flow(self) -> FlowModel:
        """The current normalising flow."""
        if not self.flows:
            flow = None
        else:
            flow = self.flows[-1]
        return flow

    def set_log_likelihood(self, func):
        """Set the log-likelihood function"""
        self.log_likelihood = func

    def get_flow(self) -> FlowModel:
        """Get a new flow object to train"""
        flow = FlowModel(config=self.flow_config, output=self.output)
        flow.initialise()
        if self.flows and not self.reset_flows:
            flow.model.load_state_dict(
                copy.deepcopy(self.flows[-1].model.state_dict())
            )
        return flow

    def to_prime(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples from the unit hypercube to samples in x'-space"""
        if self.reparam == 'logit':
            x_prime, log_j = logit(x.copy())
            log_j = log_j.sum(axis=1)
        elif self.reparam == 'gaussian_cdf':
            logger.debug('Rescaling with inverse Gaussian CDF')
            x_prime, log_j = inv_gaussian_cdf_with_log_j(x.copy())
            log_j = log_j.sum(axis=1)
        elif self.reparam is None:
            x_prime = x.copy()
            log_j = np.zeros(x.shape[0])
        else:
            raise ValueError(self.reparam)
        return x_prime, log_j

    def from_prime(self, x_prime: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples the x'-space to samples in the unit hypercube."""
        if self.reparam == 'logit':
            x, log_j = sigmoid(x_prime.copy())
            log_j = log_j.sum(axis=1)
        elif self.reparam == 'gaussian_cdf':
            logger.debug('Rescaling with Gaussian CDF')
            x, log_j = gaussian_cdf_with_log_j(x_prime.copy())
            log_j = log_j.sum(axis=1)
        elif self.reparam is None:
            x = x_prime.copy()
            log_j = np.zeros(x.shape[0])
        else:
            raise ValueError(self.reparam)
        return x, log_j

    def rescale(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert from live points."""
        x_hypercube = self.model.to_unit_hypercube(x)
        x_array = live_points_to_array(x_hypercube, self.model.names)
        x_prime, log_j = self.to_prime(x_array)
        return x_prime, log_j

    def inverse_rescale(self, x_prime: np.ndarray) -> np.ndarray:
        x_array, log_j = self.from_prime(x_prime)
        x_hypercube = numpy_array_to_live_points(x_array, self.model.names)
        x = self.model.from_unit_hypercube(x_hypercube)
        return x, log_j

    def train(
        self,
        samples: np.ndarray,
        plot: bool = False,
        output: Union[str, None] = None,
        **kwargs
    ) -> None:
        """Train the proposal with a set of samples.

        Parameters
        ----------
        samples :  numpy.ndarray
            Array of samples for training.
        plot : bool
            Flag to enable or disable plotting.
        output : Union[str, None]
            Output directory to use instead of default output. If None the
            default that was set when the class what initialised is used.
        kwargs :
            Key-word arguments passed to \
                :py:meth:`nessai.flowmodel.FlowModel.train`.
        """
        self.level_count += 1
        self.n_draws[self.level_count] = 0
        output = self.output if output is None else output
        level_output = os.path.join(
            output, f'level_{self.level_count}', ''
        )

        if not os.path.exists(level_output):
            os.makedirs(level_output, exist_ok=True)

        training_data = samples.copy()

        if plot:
            plot_live_points(
                training_data,
                filename=os.path.join(level_output, 'training_data.png')
            )

        x_prime, _ = self.rescale(training_data)
        logger.debug(
            f'Training data min and max: {x_prime.min()}, {x_prime.max()}'
        )

        if self.weighted_kl:
            logger.debug('Using weights in training')
            if self.weights_include_likelihood:
                log_weights = training_data['logW'] + training_data['logL']
            else:
                log_weights = training_data['logW']
            log_weights -= logsumexp(log_weights)
            weights = np.exp(log_weights)
            if plot:
                plot_histogram(
                    weights, filename=level_output + 'training_weights.png'
                )
        else:
            weights = None

        flow = self.get_flow()
        flow.train(
            x_prime,
            weights=weights,
            output=level_output,
            plot=plot or self.plot_training,
            **kwargs,
        )

        if plot:
            test_samples_prime, log_prob = flow.sample_and_log_prob(2000)
            test_samples, log_j_inv = self.inverse_rescale(test_samples_prime)
            log_prob -= log_j_inv
            test_samples['logG'] = log_prob
            plot_live_points(
                test_samples,
                filename=os.path.join(level_output, 'generated_samples.png')
            )
        self.flows.append(flow)
        self.levels[self.level_count] = flow

    def _compute_log_g_combined(self, x_prime, log_q, n, log_j):
        n_flows = len(self.flows)
        log_g = np.empty((x_prime.shape[0], n_flows))
        if np.isnan(x_prime).any():
            logger.warning('NaNs in samples when computing log_g')
        if not np.isfinite(x_prime).all():
            logger.warning(
                'Infinite values in the samples when computing log_g'
            )
        if log_q is not None:
            logger.debug('Setting log_g for current flow using inputs')
            logger.debug(f'n={n}')
            logger.debug(f'log_q finite={np.isfinite(log_q).all()}')
            logger.debug(f'log_j finite={np.isfinite(log_j).all()}')
            n_flows -= 1
            log_g[:, -1] = np.log(n) + log_q + log_j

        logger.debug(f'Updating log_g for flows: {list(range(n_flows))}')
        for i, flow in enumerate(self.flows[:n_flows]):
            logger.debug(f"Poolsize: {self._history['poolsize'][i]}")
            log_prob = flow.log_prob(x_prime)
            log_g[:, i] = (
                log_prob
                + log_j
                + np.log(self._history['poolsize'][i])
            )

        logger.debug(f'log_g is nan: {np.isnan(log_g).any()}')
        logger.debug(f'Initial log g: {self.initial_log_g:.2f}')
        logger.debug(
            f'Mean log g for each each flow: {log_g.mean(axis=0)}'
        )
        # Could move Jacobian here
        log_g = logsumexp(log_g, axis=1)
        if np.isnan(log_g).any():
            raise ValueError('There is a NaN in log g before initial!')
        log_g = np.logaddexp(self.initial_log_g, log_g)
        if np.isnan(log_g).any():
            raise ValueError('There is a NaN in log g!')
        return log_g

    def _compute_log_g_independent(self, x, log_q, n, log_j):
        log_g = log_q + log_j + np.log(n)
        return log_g

    def compute_log_g(
        self,
        x: np.ndarray,
        log_q: np.ndarray = None,
        n: int = None,
        log_j=None,
    ) -> np.ndarray:
        """Compute log g for an array of points.

        Parameters
        ----------
        x : np.ndarray
            Array of samples in the unit hypercube.
        """
        if self.combined_proposal:
            return self._compute_log_g_combined(x, log_q, n, log_j)
        else:
            return self._compute_log_g_independent(x, log_q, n, log_j)

    def draw(
        self,
        n: int,
        logL_min=None,
        flow_number=None
    ) -> np.ndarray:
        """Draw n new points.

        Parameters
        ----------
        n : int
            Number of points to draw.

        Returns
        -------
        np.ndarray :
            Array of new points.
        """
        if flow_number is None:
            flow_number = self.level_count
        if logL_min:
            _p = 1.1
            n = int(n)
            n_draw = int(_p * n)
        else:
            n_draw = int(1.01 * n)
        logger.debug(f'Drawing {n} points')
        # This could be managed better
        samples = np.zeros(0, dtype=self.dtype)
        n_accepted = 0
        # Remove this after testing
        # self._history['poolsize'].append(n)
        # print('You need to fix this!')
        while n_accepted < n and n_draw > 0:
            logger.debug(f'Drawing batch of {n_draw} samples')
            x_prime, log_q = \
                self.flows[flow_number].sample_and_log_prob(N=n_draw)
            assert x_prime.min() >= 0, x_prime.min()
            assert x_prime.max() <= 1, x_prime.max()
            x, log_j = self.inverse_rescale(x_prime)
            # Rescaling can sometimes produce infs that don't appear in samples
            x_check = self.rescale(x)[0]
            # Probably don't need all these checks.
            acc = (
                self.model.in_bounds(x)
                & isfinite_struct(x)
                & np.isfinite(x_check).all(axis=1)
                & np.isfinite(x_prime).all(axis=1)
                & np.isfinite(log_j)
            )
            logger.debug(f'Rejected {n_draw - acc.size} points')
            if not np.any(acc):
                continue
            x, x_prime, log_j, log_q = \
                get_subset_arrays(acc, x, x_prime, log_j, log_q)

            x['logG'] = self.compute_log_g(
                x_prime, log_q=log_q, n=n, log_j=-log_j
            )
            x['logP'] = self.model.log_prior(x)
            x['logW'] = - x['logG']
            accept = (
                np.isfinite(x['logP'])
                & np.isfinite(x['logW'])
            )
            if not np.any(accept):
                continue

            x = x[accept]
            if not np.isfinite(x['logP']).all():
                raise RuntimeError('Prior value is inf!')

            if logL_min is not None:
                raise RuntimeError('Avoid computing log-likelihood here!')
                self.log_likelihood(x)
            samples = np.concatenate([samples, x])
            if logL_min is not None:
                m = (x['logL'] >= logL_min).sum()
                n_accepted += m
            else:
                n_accepted += accept.sum()
            logger.debug(f'Accepted: {n_accepted}')

        if logL_min is not None:
            possible_idx = np.cumsum(samples['logL'] >= logL_min)
            idx = np.argmax(possible_idx >= n)
            samples = samples[:(idx + 1)]
            assert len(samples) >= n
            logger.debug(
                f"Accepted {(samples['logL'] >= logL_min).sum()} "
                f'with logL greater than {logL_min}'
            )
        else:
            entr = entropy(np.exp(samples['logG']))
            samples = samples[:n]

        entr = entropy(np.exp(samples['logG']))
        logger.info(f'Proposal self entropy: {entr:.3}')
        self._history['entropy'].append(entr)
        self._history['poolsize'].append(samples.size)

        self.draw_count += 1
        self.n_draws[self.level_count] += samples.size
        logger.debug(f'Returning {samples.size} samples')
        return samples

    def update_samples(self, samples: np.ndarray) -> None:
        """Update log W and log G in place for a set of samples.

        Parameters
        ----------
        samples : numpy.ndarray
            Array of samples to update.
        """
        x, log_j = self.rescale(samples.copy())
        new_log_g = self.compute_log_g(x, log_j=log_j)
        samples['logG'] = new_log_g
        samples['logW'] = - samples['logG']

    def _log_prior(self, x: np.ndarray) -> np.ndarray:
        """Helper function that returns the prior in the unit hyper-cube."""
        return np.zeros(x.shape[0])

    def get_proposal_log_prob(self, it: int) -> Callable:
        """Get a pointer to the function for ith proposal."""
        if it == -1:
            return self._log_prior
        elif it < len(self.flows):
            return self.flows[it].log_prob
        else:
            raise ValueError

    def compute_kl_between_proposals(
        self,
        x: np.ndarray,
        p_it: Optional[int] = None,
        q_it: Optional[int] = None,
    ) -> float:
        """Compute the KL divergence between two proposals.

        Samples should be drawn from p. If proposals aren't specified the
        current and previous proposals are used.
        """
        x_prime, log_j = self.rescale(x)
        if p_it is None:
            p_it = len(self.flows) - 1

        if q_it is None:
            q_it = len(self.flows) - 2

        if p_it == q_it:
            raise ValueError('p and q must be different')
        elif p_it < -1 or q_it < -1:
            raise ValueError(f'Invalid p_it or q_it: {p_it}, {q_it}')

        p_f = self.get_proposal_log_prob(p_it)
        q_f = self.get_proposal_log_prob(q_it)

        log_p = p_f(x_prime)
        log_q = q_f(x_prime)

        if p_it > -1:
            log_p += log_j
        if q_it > -1:
            log_q += log_j

        log_p -= logsumexp(log_p)
        log_q -= logsumexp(log_q)

        kl = np.mean(log_p - log_q)
        logger.info(f'KL between {p_it} and {q_it} is: {kl:.3}')
        return kl

    def draw_from_flows(self, n: int, weights=None) -> np.ndarray:
        """Draw n points from all flows (g).

        Parameters
        ----------
        n : int
            Number of points
        """
        logger.info(
            f'Drawing {n} samples from the combination of all the proposals'
        )
        if weights is None:
            weights = np.empty(len(self.flows) + 1)
            weights[0] = self.initial_draws
            weights[1:] = self._history['poolsize']
        weights /= np.sum(weights)
        logger.debug(f'Proposal weights: {weights}')
        a = np.random.choice(weights.size, size=n, p=weights)
        proposal_id = np.arange(weights.size)
        counts = np.bincount(a)
        prime_samples = np.empty([n, self.model.dims])
        assert len(weights) == (len(self.flows) + 1)
        count = 0
        # Draw from prior
        for i, m in zip(proposal_id, counts):
            if m == 0:
                continue
            logger.debug(f'Drawing {m} samples from the {i}th proposal.')
            if i == 0:
                prime_samples[count:(count + m)] = \
                    self.to_prime(np.random.rand(m, self.model.dims))[0]
            else:
                prime_samples[count:(count + m)] = \
                    self.flows[i-1].sample_and_log_prob(N=m)[0]
            count += m
        samples, log_j = self.inverse_rescale(prime_samples)
        finite = np.isfinite(log_j)
        samples, prime_samples, log_j = \
            get_subset_arrays(finite, samples, prime_samples, log_j)
        log_g = np.empty((samples.size, len(self.flows) + 1))
        log_g[:, 0] = np.log(counts[0])
        for i, flow in enumerate(self.flows):
            log_g[:, i + 1] = (
                flow.log_prob(prime_samples)
                - log_j
                + np.log(counts[i + 1])
            )
        log_g = logsumexp(log_g, axis=1)
        finite = np.isfinite(log_g).astype(bool)
        samples, log_g = samples[finite], log_g[finite, ...]
        logger.debug(
            f'Mean g for each each flow: {np.exp(log_g).mean(axis=0)}'
        )
        # Could move Jacobian here
        # assert np.isfinite(log_g).all()
        # log_g = self.compute_log_g(
        #     prime_samples, log_j=-log_j,
        # )
        samples['logP'] = self.model.log_prior(samples)
        samples['logG'] = log_g
        samples['logW'] = - samples['logG']
        return samples

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['log_likelihood']
        return obj