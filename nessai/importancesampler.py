# -*- coding: utf-8 -*-
"""
Importance nested sampler.
"""
import logging
import os
from timeit import default_timer as timer
from typing import Any, List, Literal, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy import optimize
from tqdm import tqdm

from . import config
from .evidence import _INSIntegralState
from .basesampler import BaseNestedSampler
from .model import Model
from .posterior import draw_posterior_samples
from .proposal.importance import ImportanceFlowProposal
from .plot import plot_1d_comparison
from .livepoint import (
    add_extra_parameters_to_live_points,
    get_dtype,
    live_points_to_dict,
    numpy_array_to_live_points,
)
from .utils.hist import auto_bins
from .utils.information import (
    differential_entropy,
    relative_entropy_from_log,
)
from .utils.optimise import optimise_meta_proposal_weights
from .utils.rescaling import logistic_function
from .utils.stats import (
    effective_sample_size,
    effective_volume,
    weighted_quantile,
)
from .utils.structures import get_subset_arrays

logger = logging.getLogger(__name__)


class ImportanceNestedSampler(BaseNestedSampler):
    """

    Parameters
    ----------
    model
        User-defined model.
    nlive
        Number of live points.
    tolerance
        Tolerance for determining when to stop the sampler.
    stopping_criterion
        Choice of stopping criterion to use.
    check_criteria
        If using multiple stopping criteria determines whether any or all
        criteria must be met.
    level_method
        Method for determining new levels.
    draw_constant
        If specified the sampler will always add a constant number of samples
        from each proposal whilst removing a variable amount. If False, the
        the number will depend on the level method chosen. Note that this will
        override the choice of live points. The number of points draw is set
        by the live points.
    min_samples
        Minimum number of samples that are used for training the next
        normalising flow.
    min_remove
        Minimum number of samples that can be removed when creating the next
        level. If less than one, the sampler will stop if the level method
        determines no samples should be removed.
    plot_likelihood_levels
        Enable or disable plotting the likelihood levels.
    trace_plot_kwargs
        Keyword arguments for the trace plot.
    """
    stopping_criterion_aliases = dict(
        dZ=['dZ', 'evidence'],
        kl=['kl'],
        dZ_ns=['dZ_ns', 'alt_evidence'],
        dZ_lp=['dZ_lp', 'evidence_lp'],
        dH=['dH', 'dH_all', 'entropy'],
        ratio=['ratio', 'ratio_all'],
        ratio_ns=['ratio_ns'],
    )
    """Dictionary of available stopping criteria and their aliases."""

    def __init__(
        self,
        model: Model,
        nlive: int = 3000,
        output: Optional[str] = None,
        seed: Optional[int] = None,
        checkpointing: bool = True,
        checkpoint_frequency: int = 5,
        resume_file: Optional[str] = None,
        plot: bool = True,
        plotting_frequency: int = 5,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        min_samples: int = 500,
        min_remove: int = 1,
        tolerance: float = np.e,
        n_update: Optional[int] = None,
        plot_pool: bool = False,
        plot_level_cdf: bool = False,
        plot_trace: bool = True,
        plot_likelihood_levels: bool = True,
        plot_training_data: bool = False,
        trace_plot_kwargs: Optional[dict] = None,
        replace_all: bool = False,
        level_method: Literal['entropy', 'quantile'] = 'entropy',
        leaky: bool = True,
        sorting: Literal['logL', 'rel_entr'] = 'logL',
        n_pool: Optional[int] = None,
        pool: Optional[Any] = None,
        stopping_criterion: str = 'ratio_ns',
        check_criteria: Literal['any', 'all'] = 'any',
        level_kwargs: Optional[dict] = None,
        annealing_target: Optional[float] = None,
        annealing_beta: Optional[float] = None,
        sigmoid_weights: bool = False,
        weighted_kl: bool = False,
        draw_constant: bool = False,
        **kwargs: Any
    ):

        self.add_fields()

        super().__init__(
            model,
            nlive,
            output=output,
            seed=seed,
            checkpointing=checkpointing,
            resume_file=resume_file,
            plot=plot,
            n_pool=n_pool,
            pool=pool,
        )

        self._posterior_samples = None
        self.initialised = False
        self.finalised = False
        self.history = None
        self.live_points_ess = np.nan
        self.sorting = sorting
        self.tolerance = None
        self.criterion = None
        self._stop_any = None
        self._current_proposal_entropy = None

        self.min_samples = min_samples
        self.min_remove = min_remove
        self.checkpoint_frequency = checkpoint_frequency
        self.n_update = n_update
        self.plot_pool = plot_pool
        self.plot_level_cdf = plot_level_cdf
        self._plot_trace = plot_trace
        self._plot_likelihood_levels = plot_likelihood_levels
        self.trace_plot_kwargs = \
            {} if trace_plot_kwargs is None else trace_plot_kwargs
        self.plot_training_data = plot_training_data
        self.plotting_frequency = plotting_frequency
        self.replace_all = replace_all
        self.leaky = leaky
        self.level_method = level_method
        self.level_kwargs = {} if level_kwargs is None else level_kwargs
        self.beta = None
        self.logX = 0.0
        self.min_logL = -np.inf
        self.logL_pre = -np.inf
        self.logL = -np.inf
        self.draw_constant = draw_constant
        self.weighted_kl = weighted_kl,
        self.sigmoid_weights = sigmoid_weights
        self.sigmoid_midpoint = None

        self.dZ = np.inf
        self.dZ_ns = np.inf
        self.dZ_lp = np.inf
        self.ratio = np.inf
        self.ratio_ns = np.inf
        self._logZ = -np.inf
        self._logZ_ns = -np.inf
        self._logZ_lp = -np.inf
        self._current_entropy = 0.0

        self._normalised_evidence = self._check_normalisation(kwargs)
        self.state = _INSIntegralState(normalised=self._normalised_evidence)

        self.final_state = None
        self.final_samples = None

        self.proposal = self.get_proposal(**kwargs)
        self.configure_iterations(min_iteration, max_iteration)

        self.configure_stopping_criterion(
            stopping_criterion,
            tolerance,
            check_criteria,
        )

        self.configure_annealing(annealing_target, annealing_beta)

        self.nested_samples = np.empty(0, dtype=get_dtype(self.model.names))
        self._log_q_ns = None
        self._log_q_lp = None

        self.update_level_time = 0.0
        self.draw_time = 0.0
        self.redraw_time = 0.0
        self.update_ns_time = 0.0
        self.update_live_points_time = 0.0
        self.add_samples_time = 0.0

        if self.replace_all:
            logger.warning('Replace all is experimental')

        if not self.leaky and self.sorting == 'rel_entr':
            raise ValueError('Invalid combination of arguments!')

        self.check_configuration()

    @property
    def log_evidence(self) -> float:
        return self.state.logZ

    @property
    def log_evidence_error(self) -> float:
        return self.state.compute_uncertainty()

    @property
    def final_log_evidence(self) -> float:
        if self.final_state:
            return self.final_state.log_evidence
        else:
            return None

    @property
    def final_log_evidence_error(self) -> float:
        if self.final_state:
            return self.final_state.log_evidence_error
        else:
            return None

    @property
    def samples_entropy(self) -> float:
        """Differential entropy of all of the samples (nested + live).

        Notes
        -----
        Compute the Monte Carlo approximation of

        .. math::
            -\\int W(x) \\log W(x) dx

        where :math:`W(x) = \\pi(x)/Q(x)`.
        """
        # Q is not normalised, so must normalise weight use meta constant
        log_p = (
            self.all_samples['logW']
            + np.log(self.proposal.normalisation_constant)
        )
        return differential_entropy(log_p)

    @property
    def current_proposal_entropy(self) -> float:
        """Differential entropy of the current proposal"""
        return self._current_proposal_entropy

    @property
    def is_checkpoint_iteration(self) -> bool:
        """Check if the sampler should checkpoint at the current iteration"""
        if self.iteration % self.checkpoint_frequency:
            return False
        else:
            return True

    @property
    def all_samples(self) -> np.ndarray:
        """Return the live points + nested samples"""
        if self.live_points is not None:
            return np.concatenate([self.nested_samples, self.live_points])
        else:
            return self.nested_samples.copy()

    @property
    def reached_tolerance(self) -> bool:
        """Indicates if tolerance has been reached.

        Checks if any or all of the criteria have been met, this depends on the
        value of :code:`check_criteria`.
        """
        if self._stop_any:
            return any(
                [c <= t for c, t in zip(self.criterion, self.tolerance)]
            )
        else:
            return all(
                [c <= t for c, t in zip(self.criterion, self.tolerance)]
            )

    @staticmethod
    def add_fields():
        """Add extra fields logW and logQ"""
        add_extra_parameters_to_live_points(['logW', 'logQ'])

    def configure_stopping_criterion(
        self,
        stopping_criterion: Union[str, List[str]],
        tolerance: Union[float, List[float]],
        check_criteria: Literal['any', 'all'],
    ) -> None:
        """Configure the stopping criterion"""
        if isinstance(stopping_criterion, str):
            stopping_criterion = [stopping_criterion]

        if isinstance(tolerance, list):
            self.tolerance = [float(t) for t in tolerance]
        else:
            self.tolerance = [float(tolerance)]

        self.stopping_criterion = []
        for c in stopping_criterion:
            for criterion, aliases in self.stopping_criterion_aliases.items():
                if c in aliases:
                    self.stopping_criterion.append(criterion)
        if not self.stopping_criterion:
            raise ValueError(
                f'Unknown stopping criterion: {stopping_criterion}'
            )
        for c, c_use in zip(stopping_criterion, self.stopping_criterion):
            if c != c_use:
                logger.info(
                    f'Stopping criterion specified ({c}) is '
                    f'an alias for {c_use}. Using {c_use}.'
                )
        if len(self.stopping_criterion) != len(self.tolerance):
            raise ValueError(
                'Number of stopping criteria must match tolerances'
            )
        self.criterion = len(self.tolerance) * [np.inf]

        logger.debug(f'Stopping criteria: {self.stopping_criterion}')
        logger.debug(f'Tolerance: {self.tolerance}')

        if check_criteria not in {'any', 'all'}:
            raise ValueError('check_criteria must be any or all')
        if check_criteria == 'any':
            self._stop_any = True
        else:
            self._stop_any = False

    def _check_normalisation(self, kwargs):
        """Check if the evidence will be correctly normalised."""
        normalised = True
        if (not self.leaky) and kwargs.get('reweight_draws', False):
            normalised = False
        return normalised

    def get_proposal(
        self,
        subdir: str = 'levels',
        **kwargs
    ):
        """Configure the proposal."""
        output = os.path.join(self.output, subdir, '')
        proposal = ImportanceFlowProposal(
            self.model,
            output,
            self.nlive,
            **kwargs
        )
        return proposal

    def configure_iterations(
        self,
        min_iteration: Optional[int],
        max_iteration: Optional[int]
    ) -> None:
        """Configure the maximum iteration."""
        if min_iteration is None:
            self.min_iteration = -1
        else:
            self.min_iteration = int(min_iteration)
        if max_iteration is None:
            self.max_iteration = np.inf
        else:
            self.max_iteration = int(max_iteration)

    def check_configuration(self) -> bool:
        """Check sampler configuration is valid.

        Returns true if all checks pass.
        """
        if self.min_samples > self.nlive:
            raise ValueError('`min_samples` must be less than `nlive`')
        if self.min_remove > self.nlive:
            raise ValueError('`min_remove` must be less than `nlive`')
        return True

    def configure_annealing(
        self,
        annealing_target: Optional[float],
        beta: Optional[float],
    ) -> None:
        """Configure likelihood annealing for training the flow.

        Parameters
        ----------
        annealing_target
            Target volume for annealing. Must be in (0, 1).
        beta
            Value for the annealing parameter beta. Must be in (0, 1). If
            annealing target is specified then this value will be used as the
            initial value for the first iteration and then overridden.
        """
        if beta is not None:
            if not (0 < beta < 1):
                raise ValueError('Annealing beta must be in (0, 1)')
            beta = float(beta)

        if annealing_target is not None:
            if not (0 < annealing_target < 1):
                raise ValueError('Annealing target must be in (0, 1)')
            self.annealing_target = float(annealing_target)
            if beta is None:
                self.beta = 1e-4
            else:
                self.beta = beta
            self.annealing = True
        elif beta is not None:
            self.annealing_target = None
            self.beta = beta
            self.annealing = True
        else:
            self.annealing = False
            self.beta = None
            self.annealing_target = None

    def _rel_entr(self, x):
        return relative_entropy_from_log(x['logL'], x['logQ'])

    def sort_points(self, x: np.ndarray, *args) -> np.ndarray:
        """Correctly sort new live points.

        Parameters
        ----------
        x
            Array to sort
        args
            Any extra iterables to sort in the same way as x.
        """
        if self.sorting == 'logL':
            idx = np.argsort(x, order='logL')
        elif self.sorting == 'rel_entr':
            idx = np.argsort(self._rel_entr(x))
        else:
            raise ValueError('Sorting much be logL or rel_entr')
        if len(args):
            return get_subset_arrays(idx, x, *args)
        else:
            return x[idx]

    def populate_live_points(self) -> None:
        """Draw the initial live points from the prior.

        The live points are automatically sorted and assigned the iteration
        number -1.
        """
        live_points = np.empty(self.nlive, dtype=get_dtype(self.model.names))
        n = 0
        logger.debug(f'Drawing {self.nlive} initial points')
        while n < self.nlive:
            points = self.model.from_unit_hypercube(
                numpy_array_to_live_points(
                    np.random.rand(self.nlive, self.model.dims),
                    self.model.names
                )
            )
            points['logP'] = self.model.log_prior(points)
            accept = np.isfinite(points['logP'])
            n_it = accept.sum()
            m = min(n_it, self.nlive - n)
            live_points[n:(n + m)] = points[accept][:m]
            n += m

        live_points['logL'] = \
            self.model.batch_evaluate_log_likelihood(live_points)

        if not np.isfinite(live_points['logL']).all():
            raise RuntimeError('Found infinite values in the log-likelihood')

        live_points['it'] = -np.ones(live_points.size)
        # Since log_Q is computed in the unit-cube
        live_points['logQ'] = np.log(self.nlive)
        live_points['logW'] = - live_points['logQ']
        self.live_points = self.sort_points(live_points)
        self._log_q_lp = np.zeros([self.nlive, 1])

    def initialise(self) -> None:
        """Initialise the nested sampler.

        Draws live points, initialises the proposal.
        """
        if self.initialised:
            logger.warning('Nested sampler has already initialised!')
        if self.live_points is None:
            self.populate_live_points()

        self.initialise_history()
        self.proposal.initialise()
        self.initialised = True

    def initialise_history(self) -> None:
        """Initialise the dictionary to store history"""
        if self.history is None:
            logger.debug('Initialising history dictionary')
            self.history = dict(
                min_logL=[],
                max_logL=[],
                logX=[],
                gradients=[],
                median_logL=[],
                leakage_live_points=[],
                leakage_new_points=[],
                logZ=[],
                logZ_ns=[],
                logZ_lp=[],
                n_added=[],
                n_removed=[],
                n_post=[],
                live_points_ess=[],
                pool_entropy=[],
                samples_entropy=[],
                proposal_entropy=[],
                likelihood_evaluations=[],
                kl_proposals=[],
                annealing_beta=[],
                stopping_criteria={
                    k: [] for k in self.stopping_criterion_aliases.keys()
                },
            )
        else:
            logger.debug('History dictionary already initialised')

    def update_history(self) -> None:
        """Update the history dictionary"""
        self.history['min_logL'].append(np.min(self.live_points['logL']))
        self.history['max_logL'].append(np.max(self.live_points['logL']))
        self.history['median_logL'].append(np.median(self.live_points['logL']))
        self.history['logX'].append(self.logX)
        self.history['gradients'].append(self.gradient)
        self.history['logZ'].append(self.state.logZ)
        self.history['logZ_ns'].append(self._logZ_ns)
        self.history['logZ_lp'].append(self._logZ_lp)
        self.history['n_post'].append(self.state.effective_n_posterior_samples)
        self.history['samples_entropy'].append(self.samples_entropy)
        self.history['proposal_entropy'].append(self.current_proposal_entropy)
        self.history['live_points_ess'].append(self.live_points_ess)
        self.history['likelihood_evaluations'].append(
            self.model.likelihood_evaluations
        )

        for k in self.stopping_criterion_aliases.keys():
            self.history['stopping_criteria'][k].append(
                getattr(self, k, np.nan)
            )

    def determine_level_quantile(self, q: float = 0.8, **kwargs) -> int:
        """Determine where the next level should be located.

        Computes the q'th quantile based on log-likelihood and log-weights.

        Parameters
        ----------
        q : float
            Quantile to use. Defaults to 0.8

        Returns
        -------
        int
            The number of live points to discard.
        """
        if self.sorting == 'logL':
            return self._determine_level_quantile_log_likelihood(q, **kwargs)
        elif self.sorting == 'rel_entr':
            return self._determine_level_quantile_rel_entr(q, **kwargs)
        else:
            raise RuntimeError('Sorting not recognised')

    def _determine_level_quantile_log_likelihood(
        self, q: float, include_likelihood: bool = False,
    ) -> int:
        logger.debug(f'Determining {q:.3f} quantile')
        a = self.live_points['logL']
        if include_likelihood:
            log_weights = self.live_points['logW'] + self.live_points['logL']
        else:
            log_weights = self.live_points['logW'].copy()
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights, dtype=np.float64)
        cutoff = weighted_quantile(a, q, weights=weights, values_sorted=True)
        n = np.argmax(a >= cutoff)
        logger.debug(f'{q:.3} quantile is logL ={cutoff:.3}')
        return int(n)

    def _determine_level_quantile_rel_entr(self, q: float) -> int:
        logger.debug(f'Determining {q:.3f} quantile')
        ess = effective_sample_size(self.live_points['logW'])
        scale = ess / self.live_points.size
        scale = 1
        logger.debug(f'Rel entr scale: {scale:.3f}')
        a = self._rel_entr(self.live_points)
        cutoff = np.quantile(a, q)
        n = np.argmax(a >= cutoff) * scale
        return int(n)

    def determine_level_entropy(
        self, q: float = 0.5,
        include_likelihood: bool = False,
        use_log_weights: bool = True,
    ) -> int:
        """Determine how many points to remove based on the entropy.

        Parameters
        ----------
        q
            Fraction by which to shrink the current level.
        include_likelihood
            Boolean to indicate whether the likelihood is included in the
            weights for each samples.
        use_log_weights
            Boolean to determine if the CDF is computed using the weights or
            log-weights.
        """
        if include_likelihood:
            log_weights = \
                self.live_points['logW'] \
                + self.live_points['logL'] \
                + np.log(self.proposal.normalisation_constant)
        else:
            log_weights = \
                self.live_points['logW'] \
                + np.log(self.proposal.normalisation_constant)
        if use_log_weights:
            p = log_weights
        else:
            p = np.exp(log_weights)
        cdf = np.cumsum(p)
        if cdf.sum() == 0:
            cdf = np.arange(len(p), dtype=float)
        cdf /= cdf[-1]
        n = np.argmax(cdf >= q)
        if self.plot and self.plot_level_cdf:
            fig = plt.figure()
            plt.plot(self.live_points['logL'], cdf)
            plt.xlabel('Log-likelihood')
            plt.title('CDF')
            plt.axhline(q, c='C1')
            plt.axvline(self.live_points['logL'][n], c='C1')
            fig.savefig(os.path.join(
                self.output, 'levels', f'level_cdf_{self.iteration}.png'
            ))
            plt.close()
        return int(n)

    def determine_level(self, method='entropy', **kwargs) -> int:
        """Determine where the next level should.

        Returns
        -------
        float :
            The log-likelihood of the quantile
        int :
            The number of samples to discard.
        """
        if method == 'quantile':
            n = self.determine_level_quantile(**kwargs)
        elif method == 'entropy':
            n = self.determine_level_entropy(**kwargs)
        else:
            raise ValueError(method)
        logger.info(f'Next level should remove {n} points')
        return n

    @staticmethod
    def get_annealing_beta(
        log_l: np.ndarray,
        log_w: np.ndarray,
        target_vol: float,
        beta0: float = 1e-3,
    ) -> float:
        """Determine the current annealing value.

        Parameters
        ----------
        log_l
            Array of log-likelihood values.
        log_w
            Array of log-weights. Must be normalised.
        target_vol
            Target volume.
        beta0
            Initial value for beta.

        Returns
        -------
        float
            Value of beta.
        """
        if not (0 < target_vol < 1):
            raise ValueError('Target volume must be in (0, 1)')

        def loss(beta):
            return np.abs(target_vol - effective_volume(log_w, log_l, beta))

        beta, _, ierr, msg = optimize.fsolve(
            loss, beta0, full_output=True
        )
        beta = float(np.clip(beta, 0.0, None))
        if ierr != 1:
            logger.warning(
                f'No solution when determining beta! Returned error: {msg}. '
                'beta0 will be used.'
            )
            beta = beta0
        logger.debug(f'Beta: {beta}')

        return beta

    def update_level(self):
        """Update the current likelihood contour"""
        st = timer()
        logger.debug('Updating the contour')
        logger.info(
            "Training data ESS: "
            f"{effective_sample_size(self.training_points['logW'])}"
        )

        if self.annealing:
            log_w = self.training_points['logW'] \
                    + np.log(self.proposal.normalisation_constant)

            if self.annealing_target:
                self.beta = self.get_annealing_beta(
                    self.training_points['logL'],
                    log_w,
                    target_vol=self.annealing_target,
                    beta0=self.beta,
                )
                self.history['annealing_beta'].append(self.beta)
            w = np.exp(log_w)
            lr = np.exp(
                self.beta
                * (
                    self.training_points['logL']
                    - self.training_points['logL'].max()
                )
            )
            weights = (w * lr) / np.sum(w)
        elif self.replace_all:
            weights = - np.exp(self.training_log_q[:, -1])
        elif self.sigmoid_weights:
            weights = logistic_function(
                (
                    self.training_points['logL']
                    - self.training_points['logL'][-1]
                ),
                x0=self.sigmoid_midpoint - self.training_points['logL'][-1],
                k=0.5,
            )
            weights += config.eps
        elif self.weighted_kl:
            log_w = self.training_points['logW'] \
                + np.log(self.proposal.normalisation_constant)
            log_w -= logsumexp(log_w)
            weights = np.exp(log_w)
        else:
            weights = None

        self.training_weights = weights

        self.proposal.train(
            self.training_points,
            plot=self.plot_training_data,
            weights=weights,
        )
        kl = self.proposal.compute_kl_between_proposals(
            self.training_points, p_it=self.iteration - 1, q_it=self.iteration,
        )
        self.history['kl_proposals'].append(kl)
        self.update_level_time += (timer() - st)

    def update_live_points(
        self,
        live_points: Optional[np.ndarray] = None,
        log_q: Optional[np.ndarray] = None,
    ) -> None:
        st = timer()
        if live_points is None:
            logger.debug('Updating existing live points')
            if self.live_points is None:
                logger.warning('No live points to update!')
                return None
            else:
                live_points = self.live_points
                log_q = self._log_q_lp
        if len(live_points) != len(log_q):
            raise ValueError('Inputs are not the same length')
        log_q = self.proposal.update_samples(live_points, log_q)
        self.update_live_points_time += (timer() - st)
        return log_q

    def update_nested_samples(self) -> None:
        """Update the nested samples to reflect the current g."""
        st = timer()
        logger.debug('Updating all nested samples')
        self._log_q_ns = \
            self.proposal.update_samples(self.nested_samples, self._log_q_ns)
        self.update_ns_time += (timer() - st)

    def draw_n_samples(self, n: int):
        """Draw n points from the proposal"""
        st = timer()
        if not self.leaky:
            logL_min = self.min_logL
        else:
            logL_min = None
        new_points, log_q = self.proposal.draw(n, logL_min=logL_min)
        if self.leaky:
            logger.info('Evaluating likelihood for new points')
            new_points['logL'] = \
                self.model.batch_evaluate_log_likelihood(new_points)
            logger.info(
                'Min. log-likelihood of new samples: '
                f"{np.min(new_points['logL'])}"
            )
            if not np.isfinite(new_points['logL']).all():
                raise ValueError('Log-likelihood contains infs')
        if np.any(np.exp(new_points['logL']) == 0):
            logger.warning('New points contain points with zero likelihood')

        self.history['leakage_new_points'].append(
            self.compute_leakage(new_points)
        )
        self.draw_time += (timer() - st)
        return new_points, log_q

    def compute_leakage(
        self, samples: np.ndarray, weights: bool = True
    ) -> float:
        """Compute the leakage for a number of samples.

        Parameters
        ----------
        samples : numpy.ndarray
            Array of samples.
        weights : bool
            If True, the weight of each sample is accounted for in the
            calculation.

        Returns
        -------
        float
            The leakage as a fraction of the total number of samples
            (or effective sample size if weights is True).
        """
        if weights:
            return (
                np.sum(samples['logW'][samples['logL'] < self.min_logL])
                / samples['logW'].sum()
            )
        else:
            return (samples['logL'] < self.min_logL).sum() / samples.size

    def add_and_update_points(self, n: int):
        """Add new points to the current set of live points.

        Parameters
        ----------
        n : int
            The number of points to add.
        """
        st = timer()
        logger.debug(f'Adding {n} points')
        new_points, log_q = self.draw_n_samples(n)
        new_points, log_q = self.sort_points(new_points, log_q)
        self._current_proposal_entropy = differential_entropy(-log_q[:, -1])
        new_points['it'] = self.iteration
        logger.info(
            "New samples ESS: "
            f"{effective_sample_size(new_points['logW'])}"
        )

        # Update the constant for computing stopping criterion or when N
        # doesn't cancel
        self.state.log_meta_constant = \
            np.log(self.proposal.normalisation_constant)

        self._log_q_lp = self.update_live_points()
        self.update_nested_samples()
        self.history['n_added'].append(new_points.size)

        if self.plot and self.plot_pool:
            plot_1d_comparison(
                self.training_points,
                new_points,
                filename=os.path.join(
                    self.output, 'levels', f'pool_{self.iteration}.png'
                )
            )
        if self.leaky:
            logger.debug('Adding all points to the live points.')
            if self.live_points is None:
                self.live_points = new_points
                self._log_q_lp = log_q
            else:
                if self.sorting == 'logL':
                    idx = np.searchsorted(
                        self.live_points['logL'], new_points['logL']
                    )
                    self.live_points = np.insert(
                        self.live_points, idx, new_points
                    )
                    self._log_q_lp = np.insert(
                        self._log_q_lp, idx, log_q, axis=0
                    )
                elif self.sorting == 'rel_entr':
                    live_points = \
                        np.concatenate([self.live_points, new_points])
                    log_q = np.concatenate([self._log_q_lp, log_q])
                    self.live_points, self.log_q = \
                        self.sort_points(live_points, log_q)
                else:
                    raise RuntimeError(
                        'Could not insert new points into existing live points'
                        f'because of unknown sorting: {self.sorting}'
                    )
        else:
            logger.debug(
                f'Only add points above logL={self.min_logL:3f} to the '
                'live points.'
            )
            cut = np.argmax(new_points['logL'] >= self.min_logL)
            self.add_to_nested_samples(new_points[:cut])
            idx = np.searchsorted(
                self.live_points['logL'], new_points[cut:]['logL']
            )
            self.live_points = np.insert(
                self.live_points, idx, new_points[cut:]
            )
            self._log_q_lp = np.insert(
                self._log_q_lp, idx, log_q[cut:], axis=0
            )
        self.live_points_ess = effective_sample_size(
            self.live_points['logW']
            + np.log(self.proposal.normalisation_constant)
        )
        self.history['leakage_live_points'].append(
            self.compute_leakage(self.live_points)
        )
        logger.info(f'Current live points ESS: {self.live_points_ess:.2f}')
        self.add_samples_time += (timer() - st)

    def add_to_nested_samples(
        self, samples: np.ndarray, log_q: Optional[np.ndarray] = None,
    ) -> None:
        """Add an array of samples to the nested samples."""
        self.nested_samples = np.concatenate([self.nested_samples, samples])
        if log_q is not None:
            if self._log_q_ns is None:
                self._log_q_ns = log_q
            else:
                self._log_q_ns = np.concatenate(
                    [self._log_q_ns, log_q], axis=0
                )

    def remove_points(self, n: int) -> None:
        """Remove points from the current set of live points.

        The evidence is updated with the discarded points.

        Parameters
        ----------
        n : int
            The number of points to remove.
        """
        if self.replace_all:
            self.history['n_removed'].append(self.live_points.size)
        else:
            self.history['n_removed'].append(n)
        logger.debug(f'Removing {n} points')

        if self.replace_all:
            self.add_to_nested_samples(self.live_points, self._log_q_lp)
            self.training_points = self.live_points[n:].copy()
            self.training_log_q = self._log_q_lp[n:].copy()
            self.live_points = None
            self._log_q_lp = None
        else:
            self.add_to_nested_samples(
                self.live_points[:n], self._log_q_lp[:n]
            )
            if self.sigmoid_weights:
                self.training_points = self.live_points.copy()
                self.sigmoid_midpoint = self.live_points['logL'][n]
            self.live_points = np.delete(self.live_points, np.s_[:n])
            self._log_q_lp = np.delete(self._log_q_lp, np.s_[:n], axis=0)
            if not self.sigmoid_weights:
                self.training_points = self.live_points.copy()

    def adjust_final_samples(self, n_batches=5):
        """Adjust the final samples"""
        orig_n_total = self.nested_samples.size
        its, counts = np.unique(self.nested_samples['it'], return_counts=True)
        assert counts.sum() == orig_n_total
        weights = counts / orig_n_total
        original_unnorm_weight = counts.copy()

        logger.debug(f'Final counts: {counts}')
        logger.debug(f'Final weights: {weights}')
        logger.debug(f'Final its: {list(self.proposal.n_requested.keys())}')

        sort_idx = np.argsort(self.nested_samples, order='it')
        samples = self.nested_samples[sort_idx].copy()
        log_q = self._log_q_ns[sort_idx].copy()
        n_total = samples.size

        # This changes the proposal because the number of samples changes
        log_evidences = np.empty(n_batches)
        log_evidence_errors = np.empty(n_batches)
        proposal = self.proposal
        for i in range(n_batches):
            new_counts = np.random.multinomial(
                orig_n_total,
                weights,
            )
            logger.debug(f'New counts: {new_counts}')
            logger.debug(new_counts.sum())

            # Draw missing samples
            for it, c, nc in zip(its, counts, new_counts):
                if nc > c:
                    logger.debug(f'Drawing {nc - c} samples from {it}')
                    if it == -1:
                        new_samples, new_log_q = \
                            proposal.draw_from_prior(nc - c)
                    else:
                        new_samples, new_log_q = \
                            proposal.draw(
                                n=(nc - c),
                                flow_number=it,
                                update_counts=False,
                            )
                    new_samples['it'] = it
                    new_samples['logL'] = \
                        self.model.batch_evaluate_log_likelihood(new_samples)
                    new_loc = np.searchsorted(samples['it'], new_samples['it'])
                    samples = np.insert(samples, new_loc, new_samples)
                    log_q = np.insert(log_q, new_loc, new_log_q, axis=0)
                    n_total = samples.size
                    counts = np.unique(samples['it'], return_counts=True)[1]
                    logger.debug(f'Updated counts: {counts}')

            idx_keep = np.zeros(n_total, dtype=bool)
            cc = 0
            for it, c, nc in zip(its, counts, new_counts):
                assert c >= nc
                idx = np.random.choice(
                    np.arange(cc, cc + c), size=nc, replace=False
                )
                idx_keep[idx] = True
                assert np.all(samples[idx]['it'] == it)
                cc += c

            batch_samples = samples[idx_keep]
            batch_log_q = log_q[idx_keep]
            assert batch_samples.size == orig_n_total

            log_Q = logsumexp(batch_log_q, b=original_unnorm_weight, axis=1)
            # Weights are normalised because the total number of samples is the
            # same.
            batch_samples['logQ'] = log_Q
            batch_samples['logW'] = -log_Q
            state = _INSIntegralState()
            state.update_evidence(batch_samples)
            log_evidences[i] = state.log_evidence
            log_evidence_errors[i] = state.log_evidence_error
            logger.debug(f'Log-evidence batch {i} = {log_evidences[i]:.3f}')

        mean_log_evidence = np.mean(log_evidences)
        standard_error = np.std(log_evidences, ddof=1)

        logger.info(f'Mean log evidence: {mean_log_evidence:.3f}')
        logger.info(f'SE log evidence: {standard_error:.3f}')
        self.adjusted_log_evidence = mean_log_evidence
        self.adjusted_log_evidence_error = standard_error

    def finalise(self) -> None:
        """Finalise the sampling process."""
        if self.finalised:
            logger.warning('Sampler already finalised')
            return
        logger.info('Finalising')

        self.add_to_nested_samples(self.live_points, self._log_q_lp)
        self.live_points = None
        self.state.update_evidence(
            self.nested_samples,
            live_points=self.live_points,
        )

        self.adjust_final_samples()

        final_kl = self.kl_divergence()
        logger.warning(
            f'Final log Z: {self.state.logZ:.3f} '
            f'+/- {self.state.compute_uncertainty():.3f}'
        )
        logger.warning(f'Final KL divergence: {final_kl:.3f}')
        logger.warning(
            f'Final ESS: {self.state.effective_n_posterior_samples:.3f}'
        )
        self.checkpoint(periodic=True)
        self.produce_plots()
        self.finalised = True

    def add_level_post_sampling(self, samples: np.ndarray, n: int) -> None:
        """Add a level to the nested sampler after initial sampling has \
            completed.
        """
        self.proposal.train(samples)
        new_samples, log_q = self.draw_n_samples(n)
        log_q = self.update_live_points(new_samples, log_q)
        self.update_nested_samples()
        self.add_to_nested_samples(new_samples)
        self.state.update_evidence_from_nested_samples(self.nested_samples)

    def compute_stopping_criterion(self) -> List[float]:
        """Compute the stopping criterion.

        The method used will depend on how the sampler was configured.
        """

        pre_logZ_ns = self._logZ_ns
        self._logZ_ns = self.state.log_evidence_nested_samples
        self.dZ_ns = np.abs(self._logZ_ns - pre_logZ_ns)

        pre_logZ_lp = self._logZ_lp
        self._logZ_lp = self.state.log_evidence_live_points
        self.dZ_lp = np.abs(self._logZ_lp - pre_logZ_lp)

        pre_logZ = self._logZ
        self._logZ = self.state.logZ
        self.dZ = np.abs(self._logZ - pre_logZ)

        self.ratio = self.state.compute_evidence_ratio(ns_only=False)
        self.ratio_ns = self.state.compute_evidence_ratio(ns_only=True)

        self.kl = self.kl_divergence(include_live_points=True)

        previous_entropy = self._current_entropy
        self._current_entropy = self.samples_entropy
        self.dH = np.abs(
            (self._current_entropy - previous_entropy) / self._current_entropy
        )
        self.ess = self.state.effective_n_posterior_samples

        logger.debug(f'Current entropy: {self._current_entropy:.3f}')
        logger.debug(f'Relative change in entropy: {self.dH:.3f}')

        cond = [getattr(self, sc) for sc in self.stopping_criterion]

        logger.info(
            f'Stopping criteria ({self.stopping_criterion}): {cond} '
            f'- Tolerance: {self.tolerance}'
        )
        return cond

    def checkpoint(self, periodic: bool = False):
        """Checkpoint the sampler."""
        if periodic is False:
            logger.warning(
                'Importance Sampler cannot checkpoint mid iteration'
            )
            return
        super().checkpoint(periodic=periodic)

    def _compute_gradient(self) -> None:
        self.logX_pre = self.logX
        self.logX = logsumexp(self.live_points['logW'])
        self.logL_pre = self.logL
        self.logL = logsumexp(
            self.live_points['logL'] - self.live_points['logQ']
        )
        self.dlogX = (self.logX - self.logX_pre)
        self.dlogL = (self.logL - self.logL_pre)
        self.gradient = self.dlogL / self.dlogX

    def nested_sampling_loop(self):
        """Main nested sampling loop."""
        self.initialise()
        logger.warning('Starting the nested sampling loop')
        if self.finalised:
            logger.warning('Sampler has already finished sampling! Aborting')
            return self.log_evidence, self.nested_samples

        while True:
            if (
                self.reached_tolerance
                and self.iteration >= self.min_iteration
            ):
                break

            self._compute_gradient()

            if self.n_update is None:
                n_remove = self.determine_level(
                    method=self.level_method, **self.level_kwargs
                )
            else:
                n_remove = self.n_update
            if n_remove == 0:
                if self.min_remove < 1:
                    logger.warning('No points to remove')
                    logger.warning('Stopping')
                    break
                else:
                    n_remove = 1
            if (self.live_points.size - n_remove) < self.min_samples:
                n_remove = self.live_points.size - self.min_samples
                logger.warning('Cannot remove all live points!')
                logger.warning(f'Removing {n_remove}')
            elif n_remove < self.min_remove:
                logger.warning(
                    f'Cannot remove less than {self.min_remove} samples'
                )
                n_remove = self.min_remove
                logger.warning(f'Removing {n_remove}')

            self.min_logL = self.live_points[n_remove]['logL'].copy()
            self.remove_points(n_remove)
            self.update_level()
            if self.draw_constant or self.replace_all:
                n_add = self.nlive
            else:
                n_add = n_remove
            self.add_and_update_points(n_add)
            self.state.update_evidence(self.nested_samples, self.live_points)
            self.iteration += 1
            self.criterion = self.compute_stopping_criterion()

            logger.warning(
                f'Update {self.iteration} - '
                f'log Z: {self.state.logZ:.3f} +/- '
                f'{self.state.compute_uncertainty():.3f} '
                f'dZ: {self.dZ:.3f} '
                f'ESS: {self.ess:.1f} '
                f"logL min: {self.live_points['logL'].min():.3f} "
                f"logL max: {self.live_points['logL'].max():.3f}"
            )
            self.update_history()
            if not self.iteration % self.plotting_frequency:
                self.produce_plots()
            if self.checkpointing and self.is_checkpoint_iteration:
                self.checkpoint(periodic=True)
            if self.iteration >= self.max_iteration:
                break

        logger.warning(
            f'Finished nested sampling loop after {self.iteration} iterations '
            f'with {self.stopping_criterion} = {self.criterion}'
        )
        self.finalise()
        logger.info(f'Level update time: {self.update_level_time}')
        logger.info(f'Log-likelihood time: {self.likelihood_evaluation_time}')
        logger.info(f'Draw time: {self.draw_time}')
        logger.info(f'Update NS time: {self.update_ns_time}')
        logger.info(f'Update live points time: {self.update_live_points_time}')
        logger.info(f'Add samples time: {self.add_samples_time}')
        return self.log_evidence, self.nested_samples

    def draw_posterior_samples(
        self,
        sampling_method: str = 'rejection_sampling',
        n: Optional[int] = None,
        use_final_samples: bool = True,
    ) -> np.ndarray:
        """Draw posterior samples from the current nested samples."""

        if use_final_samples and self.final_samples is not None:
            samples = self.final_samples
            log_w = self.final_state.log_posterior_weights
        else:
            samples = self.nested_samples
            log_w = self.state.log_posterior_weights

        posterior_samples, indices = draw_posterior_samples(
            samples,
            log_w=log_w,
            method=sampling_method,
            n=n,
            return_indices=True,
        )
        log_p = log_w[indices] - log_w[indices].max()
        h = differential_entropy(log_p)
        logger.info(f'Information in the posterior: {h:.3f} nats')
        logger.info(f'Produced {posterior_samples.size} posterior samples.')
        return posterior_samples

    def kl_divergence(self, include_live_points: bool = False) -> float:
        """Compute the KL divergence between the posterior and g"""
        if not len(self.nested_samples):
            return np.inf
        # logQ is computed on the unit hyper-cube where the prior is 1/1^n
        # so logP = 0
        log_q = self.nested_samples['logL'].copy()
        log_p = self.nested_samples['logQ'].copy()
        if include_live_points:
            log_q = np.concatenate([log_q, self.live_points['logL']])
            log_p = np.concatenate([log_p, self.live_points['logQ']])
        log_q -= logsumexp(log_q)
        log_p -= logsumexp(log_p)
        # TODO: Think about if p and q are correct.
        kl = np.mean(log_p - log_q)
        logger.info(f'KL divergence between posterior and g: {kl:.3f}')
        return float(kl)

    def draw_more_nested_samples(self, n: int) -> np.ndarray:
        """Draw more nested samples from g"""
        samples = self.proposal.draw_from_flows(n)
        samples['logL'] = self.model.batch_evaluate_log_likelihood(samples)
        state = _INSIntegralState()
        state.update_evidence(samples)
        logger.info(
            'Evidence in new nested samples: '
            f'{state.logZ:3f} +/- {state.compute_uncertainty():.3f}'
        )
        logger.info(
            'Effective number of posterior samples: '
            f'{state.effective_n_posterior_samples:3f}'
        )
        return samples

    def draw_final_samples(
        self,
        n_post: Optional[int] = None,
        n_draw: Optional[int] = None,
        max_its: int = 1000,
        max_batch_size: int = 20_000,
        max_samples_ratio: Optional[float] = 1.0,
        use_counts: bool = False,
        optimise_weights: bool = False,
        optimise_kwargs: Optional[dict] = None,
    ):
        """Draw final unbiased samples until a desired ESS is reached.

        The number of samples drawn is based on the efficiency of the existing
        nested samples up to a maximum size determined by
        :code:`max_batch_size` or on the value of :code:`n_draw. The number
        is increased by 1% to account for samples being rejected.

        Returns nested samples, NOT posterior samples.

        Restarts the multiprocessing pool for evaluations the likelihood.

        Parameters
        ----------
        n_post
            Target effective sample size for the posterior distribution. May
            not be reached if max_its is reached first. If not specified then
            the number of samples drawn will match the nested samples.
        n_draw
            Number of samples to draw from the meta proposal. Should only be
            specified if not specifying :code:`n_post`.
        max_its
            Maximum number of iterations before stopping.
        max_batch_size
            Maximum number of samples to draw in a single batch.
        max_samples_ratio
            Maximum number of samples in terms of the number of samples drawn
            during sampling. For example if :code:`max_samples=1`, up to half
            the initial number of samples will be drawn. If None, no limit is
            set.
        optimise_weights
            If True, the weights for each proposal are optimised before
            redrawing the samples.
        optimise_kwargs
            Keyword arguments passed to the optimiser function.
        use_counts
            Use the exact counts for each proposal rather than the weights.
            Not recommended. Ignored if :code:`optimise_weights` is True.

        Returns
        -------
        log_evidence
            The log evidence for the new samples
        samples
            Structured array with the new nested samples.
        """
        if n_post and n_draw:
            raise RuntimeError('Specify either `n_post` or `n_draw`')
        start_time = timer()

        if self.final_state:
            logger.warning('Existing final state will be overridden')

        # The exact counts (n_j) will not match the values used originally
        # So the evidence will need to be correctly normalised.
        self.final_state = _INSIntegralState(normalised=False)

        eff = (
            self.state.effective_n_posterior_samples
            / self.nested_samples.size
        )
        max_samples = int(max_samples_ratio * self.nested_samples.size)

        logger.debug(f'Expected efficiency: {eff:.3f}')
        if not any([n_post, n_draw]):
            n_draw = self.nested_samples.size

        if n_post:
            n_draw = int(n_post / eff)
            logger.info(f'Redrawing samples with target ESS: {n_post:.1f}')
            logger.info(f'Expect to draw approximately {n_draw:.0f} samples')
            if n_draw > max_samples:
                logger.warning(
                    f'Expected number of samples ({n_draw}) is greater than '
                    f'the maximum number of samples ({max_samples}). Final '
                    'ESS will most likely be less than the specified value.'
                )
            desc = 'ESS'
            total = int(n_post)
        else:
            desc = 'Drawing samples'
            logger.info(f'Drawing at least {n_draw} final samples')
            total = n_draw

        batch_size = int(1.05 * n_draw)
        while batch_size > max_batch_size:
            if batch_size <= 1:
                raise RuntimeError(
                    'Could not determine a valid batch size. '
                    'Consider changing the maximum batch size.'
                )
            batch_size //= 2

        logger.debug(f'Batch size: {batch_size}')

        if optimise_weights:
            if optimise_kwargs is None:
                optimise_kwargs = {}
            weights = optimise_meta_proposal_weights(
                self.nested_samples.copy(),
                self._log_q_ns.copy(),
                **optimise_kwargs,
            )
            target_counts = None
        elif use_counts:
            logger.warning('Using counts is not recommended!')
            target_counts = np.array(
                np.fromiter(self.proposal.unnormalised_weights.values(), int)
                * (batch_size / self.proposal.normalisation_constant),
                dtype=int
            )
            batch_size = target_counts.sum()
            weights = target_counts / target_counts.sum()
        else:
            weights = np.fromiter(
                self.proposal.unnormalised_weights.values(), float
            )
            weights /= weights.sum()
            target_counts = None

        n_models = self.proposal.n_proposals
        samples = np.empty([0], dtype=self.proposal.dtype)
        log_q = np.empty([0, n_models])
        counts = np.zeros(n_models)

        # Using normalised weights, so constant is zero
        log_meta_constant = 0.0
        self.final_state.log_meta_constant = log_meta_constant

        it = 0
        ess = 0

        with tqdm(
            total=total,
            desc=desc,
            unit='samples',
            bar_format=("{desc}: {percentage:.1f}%|{bar}| {n:.1f}/{total_fmt}")
        ) as pbar:
            while True:
                if n_post and (ess > n_post):
                    break
                if it >= max_its:
                    logger.warning('Reached maximum number of iterations.')
                    logger.warning('Stopping drawing final samples.')
                    break
                if n_post is None and (samples.size > n_draw):
                    break
                if max_samples_ratio and (len(samples) > max_samples):
                    logger.warning(
                        f'Reached maximum number of samples: {max_samples}'
                    )
                    logger.warning('Stopping')
                    break

                it_samples = np.empty([0], dtype=self.proposal.dtype)
                # Target counts will be None if use_counts is False
                it_samples, new_log_q, new_counts = \
                    self.proposal.draw_from_flows(
                        batch_size, counts=target_counts, weights=weights,
                    )
                log_q = np.concatenate([log_q, new_log_q], axis=0)
                counts += new_counts

                it_samples['logL'] = \
                    self.model.batch_evaluate_log_likelihood(it_samples)
                samples = np.concatenate([samples, it_samples])

                log_Q = logsumexp(log_q, b=weights, axis=1)

                if np.isposinf(log_Q).any():
                    logger.warning('Log meta proposal contains +inf')

                samples['logQ'] = log_Q
                samples['logW'] = -log_Q

                self.final_state.update_evidence(samples)
                ess = self.final_state.effective_n_posterior_samples
                if n_post:
                    pbar.n = ess
                    pbar.refresh()
                else:
                    pbar.update(it_samples.size)
                logger.debug(f'Sample count: {samples.size}')
                logger.debug(f'Current ESS: {ess}')
                it += 1

            if not n_post or (n_post and ess >= n_post):
                pbar.n = pbar.total

        logger.debug(f'Original weights: {self.proposal.unnormalised_weights}')
        logger.debug(f'New weights: {counts}')

        logger.info(f'Drew {samples.size} final samples')
        logger.info(
            f'Final evidence: {self.final_state.logZ:.3f} '
            f'+/- {self.final_state.compute_uncertainty():.3f}'
        )
        logger.info(f'Final ESS: {ess:.1f}')
        self.final_samples = samples
        self.redraw_time += (timer() - start_time)
        return self.final_state.logZ, samples

    def plot_state(
        self,
        filename: Optional[str] = None
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Produce plots with the current state of the nested sampling run.
        Plots are saved to the output directory specified at initialisation.

        Parameters
        ----------
        filename
            If specified the figure will be saved, otherwise the figure is
            returned.
        """
        n_subplots = 9
        if self.annealing_target:
            n_subplots += 1

        fig, ax = plt.subplots(n_subplots, 1, sharex=True, figsize=(15, 15))
        ax = ax.ravel()
        its = np.arange(self.iteration)

        colours = ['#4575b4', '#d73027', '#fad117', '#FF8C00']
        ls = ['-', '--', ':', '-.']

        for a in ax:
            a.vlines(self.checkpoint_iterations, 0, 1, color=colours[2])

        # Counter for each plot
        m = 0

        ax[m].plot(its, self.history['min_logL'], label='Min. Log L',
                   c=colours[0], ls=ls[0])
        ax[m].plot(its, self.history['max_logL'], label='Max. Log L',
                   c=colours[1], ls=ls[1])
        ax[m].plot(its, self.history['median_logL'], label='Median Log L',
                   c=colours[2], ls=ls[2])
        ax[m].set_ylabel('Log-likelihood')
        ax[m].legend(frameon=False)

        m += 1

        ax[m].plot(its, self.history['logX'], label='Log X')
        ax[m].set_ylabel('Log X')
        ax_gradients = plt.twinx(ax[m])
        ax_gradients.plot(
            its, self.history['gradients'], ls=ls[1], c=colours[1],
            label='Gradient'
        )
        ax_gradients.set_ylabel('dlogL/dlogX')
        handles, labels = ax[m].get_legend_handles_labels()
        handles_grad, labels_grad = ax_gradients.get_legend_handles_labels()
        ax[m].legend(
            handles + handles_grad, labels + labels_grad, frameon=False
        )

        m += 1

        ax[m].plot(
            its, self.history['logZ'], label='Log Z', c=colours[0], ls=ls[0]
        )
        ax[m].plot(
            its, self.history['logZ_ns'], label='Log Z (NS)', c=colours[1],
            ls=ls[1]
        )
        ax[m].plot(
            its, self.history['logZ_lp'], label='Log Z (LP)', c=colours[2],
            ls=ls[2]
        )
        ax[m].set_ylabel('Log-evidence')
        ax[m].legend(frameon=False)

        ax_dz = plt.twinx(ax[m])
        ax_dz.plot(its, self.history['stopping_criteria']['dZ'], label='dZ',
                   c=colours[3], ls=ls[3])
        ax_dz.set_ylabel('|dZ|')
        ax_dz.set_yscale('log')
        handles, labels = ax[m].get_legend_handles_labels()
        handles_dz, labels_dz = ax_dz.get_legend_handles_labels()
        ax[m].legend(handles + handles_dz, labels + labels_dz, frameon=False)

        m += 1

        ax[m].plot(its, self.history['likelihood_evaluations'])
        ax[m].set_ylabel('# likelihood \n evaluations')

        m += 1

        ax[m].plot(
            its, self.history['n_post'], label='Posterior',
            ls=ls[0], c=colours[0]
        )
        ax[m].plot(
            its, self.history['live_points_ess'], label='Live points',
            ls=ls[1], c=colours[1]
        )
        ax[m].set_ylabel('ESS')
        ax[m].legend(frameon=False)

        m += 1

        ax[m].plot(its, self.history['n_removed'], ls=ls[0], c=colours[0],
                   label='Removed')
        ax[m].plot(its, self.history['n_added'], ls=ls[1], c=colours[1],
                   label='Added')
        ax[m].set_ylabel('# samples')
        ax[m].legend(frameon=False)

        ax_leak = plt.twinx(ax[m])
        ax_leak.plot(its, self.history['leakage_live_points'], ls=ls[2],
                     c=colours[2], label='Total leakage')
        ax_leak.plot(its, self.history['leakage_new_points'], ls=ls[3],
                     c=colours[3], label='New leakage')
        ax_leak.set_ylabel('Leakage')
        handles, labels = ax[m].get_legend_handles_labels()
        handles_leak, labels_leak = ax_leak.get_legend_handles_labels()
        ax[m].legend(
            handles + handles_leak, labels + labels_leak, frameon=False
        )

        m += 1

        if self.annealing_target:
            ax[m].plot(
                its, self.history['annealing_beta'], c=colours[0], ls=ls[0]
            )
            ax[m].set_ylabel(r'$\beta$')
            m += 1

        ax[m].plot(
            its, self.history['samples_entropy'], c=colours[0], ls=ls[0],
            label='Overall',
        )
        ax[m].plot(
            its, self.history['proposal_entropy'], c=colours[1], ls=ls[1],
            label='Current',
        )
        ax[m].set_ylabel('Differential\n entropy')
        ax[m].legend(frameon=False)

        m += 1

        ax[m].plot(its, self.history['kl_proposals'], label='(q_i||q_i-1)',
                   c=colours[0], ls=ls[0])
        ax[m].set_ylabel('KL divergence')
        ax_kl = plt.twinx(ax[m])
        ax_kl.plot(its, self.history['stopping_criteria']['kl'],
                   label='(Q||post)', c=colours[1], ls=ls[1])
        ax_kl.set_ylabel('KL divergence')
        handles, labels = ax[m].get_legend_handles_labels()
        handles_kl, labels_kl = ax_kl.get_legend_handles_labels()
        ax[m].legend(handles + handles_kl, labels + labels_kl, frameon=False)

        m += 1

        for (i, sc), tol in zip(
            enumerate(self.stopping_criterion), self.tolerance
        ):
            ax[m].plot(
                its,
                self.history['stopping_criteria'][sc],
                label=sc,
                c=colours[i],
                ls=ls[i],
            )
            ax[m].axhline(tol, ls=':', c=colours[i])
        ax[m].legend(frameon=False)
        ax[m].set_ylabel('Stopping criterion')
        ax[m].set_yscale('log')

        ax[-1].set_xlabel('Iteration')

        fig.suptitle(f'Sampling time: {self.current_sampling_time}',
                     fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    def plot_trace(
        self,
        enable_colours: bool = True,
        filename: Optional[str] = None,
    ) -> Union[matplotlib.figure.Figure, None]:
        """Produce a trace-like plot of the nested samples.

        Parameters
        ----------
        enable_colours : bool
            If True, the iteration will be plotted on the colour axis. If
            False, the points will be plotted with a single colour.
        filename : Optional[str]
            Filename for saving the figure. If not specified the figure will
            be returned instead.

        Returns
        -------
        matplotlib.figure.Figure
            Trace plot figure. Only returned when the filename is not
            specified.
        """

        parameters = list(self.nested_samples.dtype.names)
        for p in ['logW', 'it']:
            parameters.remove(p)
        n = len(parameters)

        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(5, 2 * n))

        samples = self.all_samples

        log_w = samples['logW']

        if enable_colours:
            colour_kwargs = dict(
                c=samples['it'],
                vmin=-1,
                vmax=samples['it'].max(),
            )
        else:
            colour_kwargs = {}

        for ax, p in zip(axs, parameters):
            ax.scatter(
                log_w,
                samples[p],
                s=1.0,
                **colour_kwargs,
            )
            ax.set_ylabel(p)
        axs[-1].set_xlabel('Log W')

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    def plot_likelihood_levels(
        self,
        filename: Optional[str] = None,
        cmap: str = 'viridis',
    ) -> Optional[matplotlib.figure.Figure]:
        """Plot the distribution of the likelihood at each level.

        Parameters
        ----------
        filename
            Name of the file for saving the figure. If not specified, then
            the figure is returned.
        cmap
            Name of colourmap to use. Must be a valid colourmap in matplotlib.
        """
        samples = self.all_samples
        its = np.unique(samples['it'])
        colours = plt.get_cmap(cmap)(np.linspace(0, 1, len(its)))
        vmax = np.max(samples['logL'])
        vmin = min(
            vmax - 0.10 * np.ptp(samples['logL']),
            samples['logL'][samples['it'] == its[-1]].min()
        )

        fig, axs = plt.subplots(1, 2)
        for ax in axs:
            for it, c in zip(its, colours):
                data = samples['logL'][samples['it'] == it]
                ax.hist(
                    data,
                    auto_bins(data, max_bins=50),
                    histtype='step',
                    color=c,
                    density=True,
                    cumulative=False,
                )
            ax.set_xlabel('Log-likelihood')

        axs[0].set_ylabel('Density')
        axs[1].set_xlim(vmin, vmax)
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    def produce_plots(self, override: bool = False) -> None:
        """Produce all of the relevant plots.

        Checks if plotting is enabled.

        Parameters
        ----------
        force : bool
            Override the plotting setting and force the plots to be produced.
        """
        if self.plot or override:
            logger.debug('Producing plots')
            self.plot_state(os.path.join(self.output, 'state.png'))
            if self._plot_trace:
                self.plot_trace(
                    filename=os.path.join(self.output, 'trace.png'),
                    **self.trace_plot_kwargs,
                )
            if self._plot_likelihood_levels:
                self.plot_likelihood_levels(
                    os.path.join(self.output, 'likelihood_levels.png')
                )
        else:
            logger.debug('Skipping plots')

    def get_result_dictionary(self):
        """Get a dictionary contain the main results from the sampler."""
        d = super().get_result_dictionary()
        d['history'] = self.history
        d['nested_samples'] = live_points_to_dict(self.nested_samples)
        d['log_evidence'] = self.log_evidence
        d['log_evidence_error'] = self.log_evidence_error
        # Will all be None if the final samples haven't been drawn
        d['adjusted_log_evidence'] = self.adjusted_log_evidence
        d['adjusted_log_evidence_error'] = self.adjusted_log_evidence_error
        d['final_samples'] = (
            live_points_to_dict(self.final_samples)
            if self.final_samples is not None else None
        )
        d['final_log_evidence'] = self.final_log_evidence
        d['final_log_evidence_error'] = self.final_log_evidence_error

        d['sampling_time'] = self.sampling_time
        d['update_level_time'] = self.update_level_time
        d['add_samples_time'] = self.add_samples_time
        d['update_ns_time'] = self.update_ns_time
        d['update_live_points_time'] = self.update_live_points_time
        d['redraw_time'] = self.redraw_time

        return d

    @classmethod
    def resume(cls, filename, model, flow_config={}, weights_path=None):
        """
        Resumes the interrupted state from a checkpoint pickle file.

        Parameters
        ----------
        filename : str
            Pickle pickle to resume from
        model : :obj:`nessai.model.Model`
            User-defined model
        flow_config : dict, optional
            Dictionary for configuring the flow
        weights_path : str, optional
            Path to the weights files that will override the value stored in
            the proposal.

        Returns
        -------
        obj
            Instance of ImportanceNestedSampler
        """
        cls.add_fields()
        obj = super().resume(filename, model)
        obj.proposal.resume(model, flow_config, weights_path=weights_path)
        logger.info(f'Resuming sampler at iteration {obj.iteration}')
        logger.info(f'Current number of samples: {len(obj.nested_samples)}')
        logger.info(
            f'Current logZ: {obj.log_evidence:3f} '
            f'+/- {obj.log_evidence_error:.3f}'
        )
        logger.info(f'Current dZ: {obj.dZ:.3f}')
        return obj

    def __getstate__(self):
        d = self.__dict__
        exclude = {'model', 'proposal'}
        state = {k: d[k] for k in d.keys() - exclude}
        state['_previous_likelihood_evaluations'] = \
            d['model'].likelihood_evaluations
        state['_previous_likelihood_evaluation_time'] = \
            d['model'].likelihood_evaluation_time.total_seconds()
        return state, self.proposal

    def __setstate__(self, state):
        self.__dict__.update(state[0])
        self.proposal = state[1]
