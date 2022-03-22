# -*- coding: utf-8 -*-
"""
Importance nested sampler.
"""
import logging
import os
from timeit import default_timer as timer
from typing import Any, Literal, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from scipy.special import logsumexp
from tqdm import tqdm

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
from .utils.information import relative_entropy_from_log
from .utils.stats import effective_sample_size, weighted_quantile

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
    level_method
        Method for determining new levels.
    """

    _stopping_criterion_aliases = dict(
        dZ=['dZ', 'evidence'],
        kl=['kl'],
        dZ_ns=['dZ_ns', 'alt_evidence'],
        dZ_smc=['dZ_smc', 'smc_evidence'],
        dH=['dH', 'dH_all', 'entropy'],
        dH_lp=['dH_lp', 'lp_entropy'],
        dH_ns=['dH_ns', 'ns_entropy'],
    )

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
        min_samples: int = 1000,
        min_remove: int = 1,
        tolerance: float = 1.0,
        n_update: Optional[int] = None,
        plot_pool: bool = False,
        plot_level_cdf: bool = False,
        plot_trace: bool = True,
        replace_all: bool = False,
        update_nested_samples: bool = True,
        level_method: Literal['entropy', 'quantile'] = 'quantile',
        leaky: bool = True,
        sorting: Literal['logL', 'rel_entr'] = 'logL',
        n_pool: Optional[int] = None,
        pool: Optional[Any] = None,
        stopping_criterion: Literal['evidence', 'kl'] = 'evidence',
        min_dZ: Optional[float] = None,
        level_kwargs: Optional[dict] = None,
        annealing: bool = False,
        beta_min: float = 0.01,
        beta_max: float = 1.0,
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
        self.dZ = np.inf
        self.live_points_ess = np.nan
        self.sorting = sorting

        self.tolerance = tolerance
        self.min_samples = min_samples
        self.min_remove = min_remove
        self.criterion = np.inf
        self.checkpoint_frequency = checkpoint_frequency
        self.n_update = n_update
        self.plot_pool = plot_pool
        self.plot_level_cdf = plot_level_cdf
        self._plot_trace = plot_trace
        self.plotting_frequency = plotting_frequency
        self.replace_all = replace_all
        self._update_nested_samples = update_nested_samples
        self.leaky = leaky
        self.level_method = level_method
        self.level_kwargs = {} if level_kwargs is None else level_kwargs
        self.current_entropy = 0.0
        self.current_live_points_entropy = 0.0
        self.current_ns_entropy = 0.0
        self.dZ_smc = np.inf
        self.current_log_evidence = -np.inf
        self.annealing = annealing
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta = None
        self._initial_dZ = None
        self.logX = 0.0
        self.min_logL = -np.inf
        self.logL_pre = -np.inf
        self.logL = -np.inf

        self.min_dZ = min_dZ if min_dZ is not None else np.inf

        self._normalised_evidence = self._check_normalisation(kwargs)
        self.state = _INSIntegralState(normalised=self._normalised_evidence)

        self.final_state = None
        self.final_samples = None

        self.proposal = self.get_proposal(**kwargs)
        self.configure_iterations(min_iteration, max_iteration)

        self.configure_stopping_criterion(stopping_criterion)

        self.nested_samples = np.empty(0, dtype=get_dtype(self.model.names))

        self.update_level_time = 0.0
        self.draw_time = 0.0
        self.redraw_time = 0.0
        self.update_ns_time = 0.0
        self.update_live_points_time = 0.0
        self.add_samples_time = 0.0

        if self.replace_all:
            self._update_nested_samples = False

        if not self.leaky and self.sorting == 'rel_entr':
            raise ValueError('Invalid combination of arguments!')

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
    def live_points_entropy(self):
        log_p = self.live_points['logL'] + self.live_points['logW']
        log_p -= logsumexp(log_p)
        p = np.exp(log_p)
        return entropy(p) / np.log(p.size)

    @property
    def nested_samples_entropy(self):
        log_p = self.nested_samples['logL'] + self.nested_samples['logW']
        log_p -= logsumexp(log_p)
        p = np.exp(log_p)
        return entropy(p) / np.log(p.size)

    @property
    def all_samples_entropy(self):
        log_p = self.all_samples['logL'] + self.all_samples['logW']
        log_p -= logsumexp(log_p)
        p = np.exp(log_p)
        return entropy(p) / np.log(p.size)

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

    @staticmethod
    def add_fields():
        """Add extra fields logW and logQ"""
        add_extra_parameters_to_live_points(['logW', 'logQ'])

    def configure_stopping_criterion(self, stopping_criterion):
        """Configure the stopping criterion"""
        sc = None
        for criterion, aliases in self._stopping_criterion_aliases.items():
            if stopping_criterion in aliases:
                sc = criterion
        if sc is None:
            raise ValueError(
                f'Unknown stopping criterion: {stopping_criterion}'
            )
        self.stopping_criterion = sc
        if not stopping_criterion == sc:
            logger.info(
                f'Stopping criterion specified ({stopping_criterion}) is '
                f'an alias for {sc}. Using {sc}.'
            )

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
            combined_proposal=not self.replace_all,
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

    def _rel_entr(self, x):
        return relative_entropy_from_log(x['logL'], x['logQ'])

    def sort_points(self, x: np.ndarray) -> np.ndarray:
        """Correctly sort new live points."""
        if self.sorting == 'logL':
            x = np.sort(x, order='logL')
        elif self.sorting == 'rel_entr':
            x = x[np.argsort(self._rel_entr(x))]
        else:
            raise ValueError('Sorting much be logL or rel_entr')
        return x

    def populate_live_points(self) -> None:
        """Draw the initial live points from the prior.

        The live points are automatically sorted and assigned the iteration
        number -1.
        """
        live_points = self.model.from_unit_hypercube(
            numpy_array_to_live_points(
                np.random.rand(self.nlive, self.model.dims),
                self.model.names
            )
        )
        live_points['logL'] = \
            self.model.batch_evaluate_log_likelihood(live_points)
        live_points['it'] = -np.ones(live_points.size)
        # Since log_Q is computed in the unit-cube
        live_points['logP'] = self.model.log_prior(live_points)
        live_points['logQ'] = np.log(self.nlive)
        live_points['logW'] = - live_points['logQ']
        self.live_points = self.sort_points(live_points)

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
                n_added=[],
                n_removed=[],
                n_post=[],
                live_points_entropy=[],
                live_points_remaining_entropy=[],
                live_points_ess=[],
                pool_entropy=[],
                nested_samples_entropy=[],
                all_samples_entropy=[],
                likelihood_evaluations=[],
                max_logQ=[],
                mean_logQ=[],
                median_logQ=[],
                min_logQ=[],
                kl_proposals=[],
                beta=[],
                stopping_criteria={
                    k: [] for k in self._stopping_criterion_aliases.keys()
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
        self.history['n_post'].append(self.state.effective_n_posterior_samples)
        self.history['live_points_entropy'].append(self.live_points_entropy)
        self.history['all_samples_entropy'].append(self.all_samples_entropy)
        self.history['live_points_ess'].append(self.live_points_ess)
        self.history['live_points_remaining_entropy'].append(
            self.entropy_remaining
        )
        self.history['nested_samples_entropy'].append(
            self.nested_samples_entropy
        )
        self.history['pool_entropy'] = self.proposal.level_entropy
        self.history['likelihood_evaluations'].append(
            self.model.likelihood_evaluations
        )
        self.history['max_logQ'].append(np.max(self.live_points['logQ']))
        self.history['mean_logQ'].append(np.mean(self.live_points['logQ']))
        self.history['median_logQ'].append(
            np.median(self.live_points['logQ'])
        )
        self.history['min_logQ'].append(np.min(self.live_points['logQ']))
        self.history['beta'].append(self.beta)

        for k in self._stopping_criterion_aliases.keys():
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
            log_weights = self.live_points['logW'] + self.live_points['logL']
        else:
            log_weights = self.live_points['logW'].copy()
        if use_log_weights:
            p = log_weights
        else:
            p = np.exp(log_weights)
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        n = np.argmax(cdf >= q)
        if self.plot_level_cdf:
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

    def get_annealing_beta(self) -> float:
        """Determine the current annealing value"""
        if not self.annealing:
            beta = None
        if self.annealing in {'dZ', 'evidence', 'Z'}:
            if not np.isfinite(self.dZ):
                beta = self.beta_min
            else:
                if self._initial_dZ is None:
                    self._initial_dZ = np.log(self.dZ)
                beta = max(
                    self.beta_min,
                    (1.0 - self.beta_min)
                    * (np.log(self.dZ) - self._initial_dZ)
                    / (np.log(self.tolerance) - self._initial_dZ)
                    + self.beta_min,
                )
        else:
            beta = self.annealing
        logger.debug(f'Annealing beta = {beta:.2f}')
        return beta

    def update_level(self):
        """Update the current likelihood contour"""
        st = timer()
        logger.debug('Updating the contour')
        logger.info(
            "Training data ESS: "
            f"{effective_sample_size(self.training_points['logW'])}"
        )
        self.beta = self.get_annealing_beta()
        self.proposal.train(
            self.training_points,
            plot=self.proposal.plot_training,
            beta=self.beta,
        )
        kl = self.proposal.compute_kl_between_proposals(
            self.training_points, p_it=self.iteration - 1, q_it=self.iteration,
        )
        self.history['kl_proposals'].append(kl)
        self.update_level_time += (timer() - st)

    def update_live_points(self, live_points: np.ndarray = None):
        st = timer()
        if live_points is None:
            logger.debug('Updating existing live points')
            if self.live_points is None:
                logger.warning('No live points to update!')
                return
            else:
                live_points = self.live_points
        self.proposal.update_samples(live_points)
        self.update_live_points_time += (timer() - st)

    def update_nested_samples(self) -> None:
        """Update the nested samples to reflect the current g."""
        st = timer()
        logger.debug('Updating all nested samples')
        self.proposal.update_samples(self.nested_samples)
        self.update_ns_time += (timer() - st)

    def draw_n_samples(self, n: int):
        """Draw n points from the proposal"""
        st = timer()
        if not self.leaky:
            logL_min = self.min_logL
        else:
            logL_min = None
        new_points = self.proposal.draw(n, logL_min=logL_min)
        if self.leaky:
            logger.info('Evaluating likelihood for new points')
            new_points['logL'] = \
                self.model.batch_evaluate_log_likelihood(new_points)
        self.history['leakage_new_points'].append(
            self.compute_leakage(new_points)
        )
        self.draw_time += (timer() - st)
        return new_points

    def compute_leakage(self, samples: np.ndarray) -> float:
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
        new_points = self.draw_n_samples(n)
        new_points = self.sort_points(new_points)
        new_points['it'] = self.iteration
        logger.info(
            "New samples ESS: "
            f"{effective_sample_size(new_points['logW'])}"
        )

        if not self._normalised_evidence:
            # Update the constant to make sure the evidence is correct
            self.state.log_meta_constant = \
                np.log(self.proposal.normalisation_constant)

        self.update_live_points()
        if self._update_nested_samples:
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
            else:
                if self.sorting == 'logL':
                    idx = np.searchsorted(
                        self.live_points['logL'], new_points['logL']
                    )
                    self.live_points = np.insert(
                        self.live_points, idx, new_points
                    )
                elif self.sorting == 'rel_entr':
                    live_points = \
                        np.concatenate([self.live_points, new_points])
                    self.live_points = self.sort_points(live_points)
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
        self.live_points_ess = effective_sample_size(
            self.live_points['logW']
        )
        self.history['leakage_live_points'].append(
            self.compute_leakage(self.live_points)
        )
        logger.info(f'Current live points ESS: {self.live_points_ess:.2f}')
        self.add_samples_time += (timer() - st)

    def add_to_nested_samples(self, samples: np.ndarray) -> None:
        """Add an array of samples to the nested samples."""
        self.nested_samples = np.concatenate([self.nested_samples, samples])

    def remove_points(self, n: int) -> None:
        """Remove points from the current set of live points.

        The evidence is updated with the discarded points.

        Parameters
        ----------
        n : int
            The number of points to remove.
        """
        self.history['n_removed'].append(n)
        logger.debug(f'Removing {n} points')
        self.add_to_nested_samples(self.live_points[:n])
        if self._update_nested_samples:
            self.state.update_evidence_from_nested_samples(
                self.nested_samples
            )
        else:
            self.state.update_evidence(self.live_points[:n])
        if self.replace_all:
            self.training_points = self.live_points[n:].copy()
            self.live_points = None
        else:
            self.live_points = np.delete(self.live_points, np.s_[:n])
            self.training_points = self.live_points.copy()
        self.entropy_remaining = entropy(
            np.exp(self.training_points['logW'])
        )

    def finalise(self) -> None:
        """Finalise the sampling process."""
        if self.finalised:
            logger.warning('Sampler already finalised')
            return
        logger.info('Finalising')
        self.add_to_nested_samples(self.live_points)
        self.state.update_evidence_from_nested_samples(self.nested_samples)
        self.live_points = None
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
        new_samples = self.draw_n_samples(n)
        self.update_live_points(new_samples)
        if self._update_nested_samples:
            self.update_nested_samples(n)
        self.add_to_nested_samples(new_samples)
        self.state.update_evidence_from_nested_samples(self.nested_samples)

    def compute_stopping_criterion(self) -> float:
        """Compute the stopping criterion.

        The method used will depend on how the sampler was configured.
        """
        # Version for SMC_NS
        previous_log_evidence = self.current_log_evidence
        log_Z_with_live_points = \
            self.state.compute_updated_log_Z(self.live_points)

        self.dZ_smc = np.abs(
            log_Z_with_live_points - previous_log_evidence
        )

        current_ln_Z = self.state.logZ
        self.dZ_ns = np.abs(current_ln_Z - self.initial_ln_Z)
        self.dZ = self.state.compute_condition(self.live_points)
        logger.debug(f'dZ_NS: {self.dZ_ns}')
        logger.debug(f'dZ_smc: {self.dZ_smc}')
        self.kl = self.kl_divergence(include_live_points=True)

        previous_entropy = self.current_entropy
        previous_live_points_entropy = self.current_live_points_entropy
        previous_ns_entropy = self.current_ns_entropy
        self.current_entropy = self.all_samples_entropy
        self.current_live_points_entropy = self.live_points_entropy
        self.current_ns_entropy = self.nested_samples_entropy
        self.dH = np.abs(
            (self.current_entropy - previous_entropy) / self.current_entropy
        )
        self.dH_lp = np.abs(
            (self.current_live_points_entropy - previous_live_points_entropy)
            / self.current_live_points_entropy
        )
        self.dH_ns = np.abs(
            (self.current_ns_entropy - previous_ns_entropy)
            / self.current_entropy
        )

        self.current_log_evidence = \
            self.state.compute_updated_log_Z(self.live_points)

        logger.debug(f'Current entropy: {self.current_entropy:.3f}')
        logger.debug(f'Relative change in entropy: {self.dH:.3f}')
        logger.debug(
            f'Current LP entropy: {self.current_live_points_entropy:.3f}'
        )
        logger.debug(
            f'Relative change in LP entropy: {self.dH_lp:.3f}'
        )
        logger.debug(
            f'Current NS entropy: {self.current_ns_entropy:.3f}'
        )
        logger.debug(
            f'Relative change in NS entropy: {self.dH_ns:.3f}'
        )

        cond = getattr(self, self.stopping_criterion)

        logger.info(
            f'Stopping criterion: {cond:.3f} - Tolerance: {self.tolerance:.3f}'
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
                self.criterion <= self.tolerance
                and self.iteration >= self.min_iteration
            ):
                if self.dZ <= self.min_dZ:
                    logger.debug('Stopping')
                    break

            self._compute_gradient()

            self.initial_ln_Z = self.state.logZ
            if self.n_update is None:
                n_remove = self.determine_level(
                    method=self.level_method, **self.level_kwargs
                )
            else:
                n_remove = self.n_update
            if n_remove == 0:
                logger.warning('No points to remove')
                break
            if (self.live_points.size - n_remove) < self.min_samples:
                n_remove = self.live_points.size - self.min_samples
                logger.critical('Cannot remove all live points!')
                logger.critical(f'Removing {n_remove}')
            elif n_remove < self.min_remove:
                logger.critical(
                    f'Cannot remove less than {self.min_remove} samples'
                )
                n_remove = self.min_remove
                logger.critical(f'Removing {n_remove}')

            self.min_logL = self.live_points[n_remove]['logL'].copy()
            self.remove_points(n_remove)
            self.update_level()
            if self.replace_all:
                n_add = self.nlive
            else:
                n_add = n_remove
            self.add_and_update_points(n_add)
            self.iteration += 1
            self.criterion = self.compute_stopping_criterion()
            logger.info(f'Live points entropy: {self.live_points_entropy}')
            logger.info(f'NS entropy: {self.nested_samples_entropy}')
            logger.warning(
                f'Update {self.iteration} - '
                f'log Z: {self.state.logZ:.3f} +/- '
                f'{self.state.compute_uncertainty():.3f} '
                f'dZ: {self.state.compute_condition(self.live_points):.3f} '
                f'H: {self.entropy_remaining:.3f} '
                f'ESS: {self.state.effective_n_posterior_samples:.1f} '
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
            f'with dZ = {self.dZ:.3f}'
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

        posterior_samples = draw_posterior_samples(
            samples,
            log_w=log_w,
            method=sampling_method,
            n=n,
        )
        H = entropy(np.exp(log_w))
        logger.info(f'Information in the posterior: {H:.3f} nats')
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
        state.update_evidence_from_nested_samples(samples)
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
        max_its: int = 10,
        max_batch_size: int = 100_000,
    ):
        """Draw final unbiased samples until a desired ESS is reached.

        The number of samples drawn is based on the efficiency of the existing
        nested samples up to a maximum size determined by
        :code:`max_batch_size` or on the value of :code:`n_draw. The number
        is increased by 1% to account for samples being rejected.

        Returns nested samples, NOT posterior samples.

        Restarts the multiprocessing pool for evaluatins the likelihood.

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
            logger.warning('Existing final state will be overriden')
        self.final_state = _INSIntegralState()

        eff = (
            self.state.effective_n_posterior_samples
            / self.nested_samples.size
        )

        logger.debug(f'Expected efficiency: {eff:.3f}')
        if not any([n_post, n_draw]):
            n_draw = self.nested_samples.size

        if n_post:
            n_draw = n_post / eff
            logger.info(f'Redrawing samples with target ESS: {n_post:.1f}')
            logger.info(f'Expect to draw approximately {n_draw:.0f} samples')
            desc = 'ESS'
            total = int(n_post)
        else:
            desc = 'Drawing samples'
            logger.info(f'Drawing at least {n_draw} final samples')
            total = n_draw

        batch_size = int(1.01 * n_draw)
        while batch_size > max_batch_size:
            if batch_size <= 1:
                raise RuntimeError(
                    'Could not determine a valid batch size. '
                    'Consider changing the maximum batch size.'
                )
            batch_size //= 2

        logger.debug(f'Batch size: {batch_size}')
        target_counts = np.array(
            np.fromiter(self.proposal.unnormalised_weights.values(), int)
            * (batch_size / self.proposal.normalisation_constant),
            dtype=int
        )
        batch_size = target_counts.sum()
        n_models = self.proposal.n_proposals
        samples = np.empty([0], dtype=self.proposal.dtype)
        log_q = np.empty([0, n_models])
        counts = np.zeros(n_models)
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

                # Draw all samples before computing the likelihood
                it_samples = np.empty([0], dtype=self.proposal.dtype)
                for _ in range(int(np.ceil(n_draw / batch_size))):
                    new_samples, new_log_q, new_counts = \
                        self.proposal.draw_from_flows(
                            batch_size, counts=target_counts
                        )
                    it_samples = np.concatenate([it_samples, new_samples])
                    log_q = np.concatenate([log_q, new_log_q], axis=0)
                    counts += new_counts

                it_samples['logL'] = \
                    self.model.batch_evaluate_log_likelihood(it_samples)
                samples = np.concatenate([samples, it_samples])

                log_Q = logsumexp(log_q + np.log(counts), axis=1)
                # Reject samples with infs
                samples = samples[np.isfinite(log_Q)]

                samples['logQ'] = log_Q
                samples['logW'] = -log_Q

                self.final_state.update_evidence_from_nested_samples(samples)
                ess = self.final_state.effective_n_posterior_samples
                if n_post:
                    pbar.n = ess
                    pbar.refresh()
                else:
                    pbar.update(it_samples.size)
                logger.debug(f'Sample count: {samples.size}')
                logger.debug(f'Current ESS: {ess}')
                it += 1

            pbar.n = pbar.total

        logger.debug(f'Original weights: {self.proposal.unnormalised_weights}')
        logger.debug(f'New weights: {counts}')

        logger.info(f'Drew {samples.size} final samples')
        logger.info(
            f'Final evidence: {self.final_state.logZ:.3f} '
            f'+/- {self.final_state.compute_uncertainty():.3f}'
        )
        logger.info(f'Final ess: {ess:.1f}')
        self.final_samples = samples
        self.redraw_time += (timer() - start_time)
        return self.final_state.logZ, samples

    def plot_state(
        self,
        filename: Optional[str] = None
    ) -> Optional[plt.figure]:
        """
        Produce plots with the current state of the nested sampling run.
        Plots are saved to the output directory specified at initialisation.

        Parameters
        ----------
        filename
            If specifie the figure will be saved, otherwise the figure is
            returned.
        """
        fig, ax = plt.subplots(9, 1, sharex=True, figsize=(15, 15))
        ax = ax.ravel()
        its = np.arange(self.iteration)

        colours = ['#4575b4', '#d73027', '#fad117', '#FF8C00']
        ls = ['-', '--', ':', '-.']

        for a in ax:
            a.vlines(self.checkpoint_iterations, 0, 1, color=colours[2])

        # Counter for each plot
        m = 0

        ax[m].plot(its, self.history['min_logL'], label='Min logL',
                   c=colours[0], ls=ls[0])
        ax[m].plot(its, self.history['max_logL'], label='Max logL',
                   c=colours[1], ls=ls[1])
        ax[m].plot(its, self.history['median_logL'], label='Median logL',
                   c=colours[2], ls=ls[2])
        ax[m].set_ylabel('Log-likelihood')
        ax[m].legend(frameon=False)

        m += 1

        ax[m].plot(its, self.history['logX'])
        ax[m].set_ylabel('log X')
        ax_gradients = plt.twinx(ax[m])
        ax_gradients.plot(
            its, self.history['gradients'], ls=ls[1], c=colours[1]
        )
        ax_gradients.set_ylabel('dlogL/dlogX')

        m += 1

        ax[m].plot(its, self.history['logZ'], label='logZ', c=colours[0],
                   ls=ls[0])
        ax[m].set_ylabel('Log-evidence')
        ax[m].legend(frameon=False)

        ax_dz = plt.twinx(ax[m])
        ax_dz.plot(its, self.history['stopping_criteria']['dZ'], label='dZ',
                   c=colours[1], ls=ls[1])
        ax_dz.set_ylabel('dZ')
        ax_dz.set_yscale('log')
        ax_dz.axhline(self.tolerance, label=f'dZ={self.tolerance}', ls=':',
                      c=colours[2])
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
                     c=colours[2])
        ax_leak.plot(its, self.history['leakage_new_points'], ls=ls[3],
                     c=colours[3])
        ax_leak.set_ylabel('Leakage')

        m += 1

        ax[m].plot(its, self.history['nested_samples_entropy'], c=colours[0],
                   ls=ls[0], label='Nested samples')
        ax[m].plot(its, self.history['live_points_entropy'], c=colours[1],
                   ls=ls[1], label='Live points')
        ax[m].plot(its, self.history['all_samples_entropy'], c=colours[2],
                   ls=ls[2], label='Combined')
        ax[m].legend(frameon=False)
        ax[m].set_ylabel('Normalised entropy')

        m += 1

        ax[m].plot(its, self.history['kl_proposals'], label='(q_i||q_i-1)',
                   c=colours[0], ls=ls[0])
        ax[m].set_ylabel('KL divergence')
        ax_kl = plt.twinx(ax[m])
        ax_kl.plot(its, self.history['stopping_criteria']['kl'],
                   label='(g||post)', c=colours[1], ls=ls[1])
        ax_kl.set_ylabel('KL divergence')
        handles, labels = ax[m].get_legend_handles_labels()
        handles_kl, labels_kl = ax_kl.get_legend_handles_labels()
        ax[m].legend(handles + handles_kl, labels + labels_kl, frameon=False)

        m += 1
        ax[m].plot(
            its,
            self.history['stopping_criteria'][self.stopping_criterion],
            label=self.stopping_criterion,
            c=colours[0],
            ls=ls[0],
        )
        ax[m].axhline(self.tolerance, label='Threshold', ls='-', c='grey')
        ax[m].legend(frameon=False)
        ax[m].set_ylabel('Stopping criteria')
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
        self, filename: Optional[str] = None
    ) -> Union[plt.figure, None]:
        """Produce a trace-like plot of the nested samples."""

        parameters = list(self.nested_samples.dtype.names)
        for p in ['logW', 'it']:
            parameters.remove(p)
        n = len(parameters)

        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(5, 2 * n))

        log_w = self.nested_samples['logW']

        for ax, p in zip(axs, parameters):
            ax.scatter(
                log_w,
                self.nested_samples[p],
                c=self.nested_samples['it'],
                s=1.0,
                vmin=-1,
                vmax=self.nested_samples['it'].max(),
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
        ---------
        force : bool
            Override the plotting setting and force the plots to be produced.
        """
        if self.plot or override:
            logger.debug('Producing plots')
            self.plot_state(os.path.join(self.output, 'state.png'))
            if self._plot_trace:
                self.plot_trace(os.path.join(self.output, 'trace.png'))
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
