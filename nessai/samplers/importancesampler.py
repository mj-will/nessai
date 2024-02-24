# -*- coding: utf-8 -*-
"""
Importance nested sampler.
"""
import datetime
import logging
import os
from typing import Any, Callable, List, Literal, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from .base import BaseNestedSampler
from .. import config
from ..evidence import _INSIntegralState, log_evidence_from_ins_samples
from ..model import Model
from ..posterior import draw_posterior_samples
from ..proposal.importance import ImportanceFlowProposal
from ..plot import nessai_style, plot_1d_comparison
from ..livepoint import (
    add_extra_parameters_to_live_points,
    get_dtype,
)
from ..utils.hist import auto_bins
from ..utils.information import differential_entropy
from ..utils.optimise import optimise_meta_proposal_weights
from ..utils.stats import (
    effective_sample_size,
    weighted_quantile,
)
from ..utils.structures import get_subset_arrays, get_inverse_indices

logger = logging.getLogger(__name__)


class OrderedSamples:
    """Samples ordered by log-likelihood.

    Parameters
    ----------
    strict_threshold
        If true, when adding new samples, only those above the current
        log-likelihood threshold will be added to the live points.
    replace_all
        If true, all samples will be remove when calling :code:`remove_samples`
    save_log_q
        If true, :code:`log_q` will be saved when the instance is pickled. This
        makes resuming faster but increases the disk usage. If false, the
        values will not be saved and must be recomputed.
    """

    def __init__(
        self,
        strict_threshold: bool = False,
        replace_all: bool = False,
        save_log_q: bool = False,
    ) -> None:
        self.samples = None
        self.log_q = None
        self.live_points_indices = None
        self.nested_samples_indices = np.empty(0, dtype=int)
        self.strict_threshold = strict_threshold
        self.replace_all = replace_all
        self.state = _INSIntegralState()
        self.log_likelihood_threshold = None
        self.save_log_q = save_log_q

    @property
    def live_points(self) -> np.ndarray:
        """The current set of live points"""
        if self.live_points_indices is None:
            return None
        else:
            return self.samples[self.live_points_indices]

    @live_points.setter
    def live_points(self, value):
        if value is not None:
            raise ValueError("Can only set live points to None!")
        self.live_points_indices = None

    @property
    def nested_samples(self) -> np.ndarray:
        """The current set of discarded points"""
        if self.nested_samples_indices is None:
            return None
        else:
            return self.samples[self.nested_samples_indices]

    def update_log_likelihood_threshold(self, threshold: float) -> None:
        """Update the log-likelihood threshold.

        Only relevant if using :code:`strict_threshold=True`.

        Parameters
        ----------
        threshold : float
            New log-likelihood threshold
        """
        self.log_likelihood_threshold = threshold

    def sort_samples(self, samples: np.ndarray, *args) -> np.ndarray:
        """Correctly sort new live points.

        Parameters
        ----------
        x
            Array to sort
        args
            Any extra iterables to sort in the same way as x.
        """
        idx = np.argsort(samples, order="logL")
        if len(args):
            return get_subset_arrays(idx, samples, *args)
        else:
            return samples[idx]

    def add_initial_samples(
        self, samples: np.ndarray, log_q: np.ndarray
    ) -> None:
        self.samples, self.log_q = self.sort_samples(samples, log_q)
        self.live_points_indices = np.arange(self.samples.size, dtype=int)

    def add_samples(self, samples: np.ndarray, log_q: np.ndarray) -> None:
        """Add samples the existing samples"""
        # Insert samples into existing samples
        samples, log_q = self.sort_samples(samples, log_q)
        indices = np.searchsorted(self.samples["logL"], samples["logL"])
        self.samples = np.insert(self.samples, indices, samples)
        self.log_q = np.insert(self.log_q, indices, log_q, axis=0)

        if self.strict_threshold:
            n = np.argmax(
                self.samples["logL"] >= self.log_likelihood_threshold
            )
            indices = np.arange(len(self.samples))
            self.nested_samples_indices = indices[:n]
            self.live_points_indices = indices[n:]
        else:
            # Indices after insertion are indices + n before
            new_indices = indices + np.arange(len(indices))

            # Indices of all previous samples
            old_indices = get_inverse_indices(self.samples.size, new_indices)

            if len(old_indices) != (self.samples.size - samples.size):
                raise RuntimeError("Mismatch in updated_indices!")

            # Updated indices of nested samples
            self.nested_samples_indices = old_indices[
                self.nested_samples_indices
            ]

            if self.live_points_indices is None:
                self.live_points_indices = new_indices
            else:
                self.live_points_indices = old_indices[
                    self.live_points_indices
                ]
                insert_indices = np.searchsorted(
                    self.live_points_indices, new_indices
                )
                self.live_points_indices = np.insert(
                    self.live_points_indices,
                    insert_indices,
                    new_indices,
                )

    def add_to_nested_samples(self, indices: np.ndarray) -> None:
        """Move a set of samples from the live points to the nested samples"""
        sort_indices = np.searchsorted(self.nested_samples_indices, indices)
        self.nested_samples_indices = np.insert(
            self.nested_samples_indices,
            sort_indices,
            indices,
        )

    def remove_samples(self) -> int:
        """Remove samples all samples below the current log-likelihood
        threshold.

        Returns
        -------
        n : int
            The number of samples removed
        """
        if self.replace_all:
            n = len(self.live_points_indices)
            self.add_to_nested_samples(self.live_points_indices)
            self.live_points_indices = None
        else:
            n = np.argmax(
                self.live_points["logL"] >= self.log_likelihood_threshold
            )
            self.add_to_nested_samples(self.live_points_indices[:n])
            self.live_points_indices = np.delete(
                self.live_points_indices, np.s_[:n]
            )
        return n

    def update_evidence(self) -> None:
        """Update the evidence estimate given the current samples"""
        self.state.update_evidence(
            nested_samples=self.nested_samples,
            live_points=self.live_points,
        )

    def finalise(self) -> None:
        """Finalise the samples by consuming all of the live points"""
        self.add_to_nested_samples(self.live_points_indices)
        self.live_points = None
        self.state.update_evidence(self.samples)

    def compute_importance(self, importance_ratio: float = 0.5):
        """Compute the importance

        Parameters
        ----------
        importance_factor :
            Relative importance of the posterior versus the evidence. 1 is
            only the posterior and 0 is only the evidence.

        Returns
        -------
        dict
            Dictionary containing the total, posterior and evidence importance
            as a function of iteration.
        """
        log_imp_post = -np.inf * np.ones(self.log_q.shape[1])
        log_imp_z = -np.inf * np.ones(self.log_q.shape[1])
        for i, it in enumerate(range(-1, self.log_q.shape[-1] - 1)):
            sidx = np.where(self.samples["it"] == it)[0]
            zidx = np.where(self.samples["it"] >= it)[0]
            if len(sidx):
                log_imp_post[i] = logsumexp(
                    self.samples["logL"][sidx] + self.samples["logW"][sidx]
                ) - np.log(len(sidx))
            if len(zidx):
                log_imp_z[i] = logsumexp(
                    self.samples["logL"][zidx] + self.samples["logW"][zidx]
                ) - np.log(len(zidx))
        imp_z = np.exp(log_imp_z - logsumexp(log_imp_z))
        imp_post = np.exp(log_imp_post - logsumexp(log_imp_post))
        imp = (1 - importance_ratio) * imp_z + importance_ratio * imp_post
        return {"total": imp, "posterior": imp_post, "evidence": imp_z}

    def compute_evidence_ratio(
        self, threshold: Optional[float] = None
    ) -> float:
        """Compute the evidence ratio given the current log-likelihood
        threshold.

        Parameters
        ----------
        threshold
            Log-likelihood threshold to use instead of the current value.
        """
        if threshold is None:
            threshold = self.log_likelihood_threshold
        above_threshold = self.samples["logL"] >= threshold
        log_z_above = log_evidence_from_ins_samples(
            self.samples[above_threshold]
        )
        return log_z_above - self.state.log_evidence

    def __getstate__(self):
        d = self.__dict__
        exclude = {"log_q"}
        state = {k: d[k] for k in d.keys() - exclude}
        if self.save_log_q:
            state["log_q"] = self.log_q
        else:
            state["log_q"] = None
        return state


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
    threshold_method
        Method for determining new likelihood threshold.
    threshold_kwargs
        Keyword arguments for function that determines the likelihood
        threshold.
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
    max_samples
        The maximum number of samples at any given iteration. If not specified,
        then there is no limit to the number of samples.
    plot_likelihood_levels
        Enable or disable plotting the likelihood levels.
    trace_plot_kwargs
        Keyword arguments for the trace plot.
    strict_threshold : bool
        If true, when drawing new samples, only those with likelihoods above
        the current threshold will be added to the live points. If false, all
        new samples are added to the live points.
    save_log_q : bool
        Boolean that determines if the log_q array is saved when checkpointing.
        If False, this can help reduce the disk usage.
    """

    stopping_criterion_aliases = dict(
        ratio=["ratio", "ratio_all"],
        ratio_ns=["ratio_ns"],
        Z_err=["Z_err", "evidence_error"],
        log_dZ=["log_dZ", "log_evidence"],
        ess=[
            "ess",
        ],
    )
    """Dictionary of available stopping criteria and their aliases."""

    def __init__(
        self,
        model: Model,
        nlive: int = 5000,
        n_initial: Optional[int] = None,
        output: Optional[str] = None,
        seed: Optional[int] = None,
        checkpointing: bool = True,
        checkpoint_interval: int = 600,
        checkpoint_on_iteration: bool = False,
        checkpoint_callback: Optional[Callable] = None,
        save_existing_checkpoint: bool = False,
        save_log_q: bool = False,
        logging_interval: int = None,
        log_on_iteration: bool = True,
        resume_file: Optional[str] = None,
        plot: bool = True,
        plotting_frequency: int = 5,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        min_samples: int = 500,
        min_remove: int = 1,
        max_samples: Optional[int] = None,
        stopping_criterion: str = "ratio",
        tolerance: float = 0.0,
        n_update: Optional[int] = None,
        plot_pool: bool = False,
        plot_level_cdf: bool = False,
        plot_trace: bool = True,
        plot_likelihood_levels: bool = True,
        plot_training_data: bool = False,
        plot_extra_state: bool = False,
        trace_plot_kwargs: Optional[dict] = None,
        replace_all: bool = False,
        threshold_method: Literal["entropy", "quantile"] = "entropy",
        threshold_kwargs: Optional[dict] = None,
        n_pool: Optional[int] = None,
        pool: Optional[Any] = None,
        check_criteria: Literal["any", "all"] = "any",
        weighted_kl: bool = False,
        draw_constant: bool = True,
        train_final_flow: bool = False,
        bootstrap: bool = False,
        close_pool: bool = False,
        strict_threshold: bool = False,
        draw_iid_live: bool = True,
        **kwargs: Any,
    ):
        self.add_fields()

        super().__init__(
            model,
            nlive,
            output=output,
            seed=seed,
            checkpointing=checkpointing,
            checkpoint_interval=checkpoint_interval,
            checkpoint_on_iteration=checkpoint_on_iteration,
            checkpoint_callback=checkpoint_callback,
            logging_interval=logging_interval,
            log_on_iteration=log_on_iteration,
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
        self.tolerance = None
        self.criterion = None
        self._stop_any = None
        self._current_proposal_entropy = None
        self.importance = dict(total=None, posterior=None, evidence=None)

        self.draw_iid_live = draw_iid_live

        self.n_initial = self.nlive if n_initial is None else n_initial
        self.min_samples = min_samples
        self.min_remove = min_remove
        self.max_samples = max_samples
        self.n_update = n_update
        self.plot_pool = plot_pool
        self._plot_level_cdf = plot_level_cdf
        self._plot_trace = plot_trace
        self._plot_likelihood_levels = plot_likelihood_levels
        self._plot_extra_state = plot_extra_state
        self.trace_plot_kwargs = (
            {} if trace_plot_kwargs is None else trace_plot_kwargs
        )
        self.plot_training_data = plot_training_data
        self.plotting_frequency = plotting_frequency
        self.replace_all = replace_all
        self.threshold_method = threshold_method
        self.threshold_kwargs = (
            {} if threshold_kwargs is None else threshold_kwargs
        )
        self.strict_threshold = strict_threshold
        self.logX = 0.0
        self.log_likelihood_threshold = -np.inf
        self.logL_pre = -np.inf
        self.logL = -np.inf
        self.draw_constant = draw_constant
        self._train_final_flow = train_final_flow
        self.bootstrap = bootstrap
        self.bootstrap_log_evidence = None
        self.bootstrap_log_evidence_error = None
        self.weighted_kl = weighted_kl
        self.save_existing_checkpoint = save_existing_checkpoint
        self.save_log_q = save_log_q

        self.log_dZ = np.inf
        self.ratio = np.inf
        self.ratio_ns = np.inf
        self.ess = 0.0
        self.Z_err = np.inf

        self._final_samples = None

        self.proposal = self.get_proposal(**kwargs)
        self.configure_iterations(min_iteration, max_iteration)

        self.configure_stopping_criterion(
            stopping_criterion,
            tolerance,
            check_criteria,
        )

        self.training_samples = OrderedSamples(
            strict_threshold=self.strict_threshold,
            replace_all=self.replace_all,
            save_log_q=self.save_log_q,
        )
        if self.draw_iid_live:
            self.iid_samples = OrderedSamples(
                strict_threshold=self.strict_threshold,
                replace_all=self.replace_all,
                save_log_q=self.save_log_q,
            )
        else:
            self.iid_samples = None

        self.training_time = datetime.timedelta()
        self.draw_samples_time = datetime.timedelta()
        self.add_and_update_samples_time = datetime.timedelta()
        self.draw_final_samples_time = datetime.timedelta()

        if self.replace_all:
            logger.warning("Replace all is experimental")

        if close_pool:
            logger.critical(
                "ImportanceNestedSampler will NOT close the multiprocessing "
                "pool automatically. This must be done manually."
            )

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
    def final_log_posterior_weights(self) -> float:
        if self.final_state:
            return self.final_state.log_posterior_weights
        else:
            return None

    @property
    def posterior_effective_sample_size(self) -> float:
        """The effective sample size of the posterior distribution.

        Returns the value for the posterior samples from the resampling step if
        they are available, otherwise falls back to the samples from the
        initial sampling.
        """
        if self.final_state:
            return self.final_state.effective_n_posterior_samples
        else:
            return self.state.effective_n_posterior_samples

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
        return differential_entropy(self.samples_unit["logW"])

    @property
    def current_proposal_entropy(self) -> float:
        """Differential entropy of the current proposal"""
        return self._current_proposal_entropy

    @property
    def _ordered_samples(self) -> OrderedSamples:
        """The set of ordered samples to treat as the 'main' samples.

        If :code:`draw_iid_live=True` these will be the i.i.d samples. If it
        is False, then it will be the training samples.
        """
        if self.draw_iid_live:
            return self.iid_samples
        else:
            return self.training_samples

    @property
    def samples_unit(self) -> np.ndarray:
        return self._ordered_samples.samples

    @property
    def samples(self) -> np.ndarray:
        return self.model.from_unit_hypercube(self.samples_unit)

    @property
    def log_q(self) -> np.ndarray:
        return self._ordered_samples.log_q

    @property
    def live_points_unit(self) -> np.ndarray:
        """The current set of live points"""
        return self._ordered_samples.live_points

    @live_points_unit.setter
    def live_points_unit(self, samples) -> None:
        if samples is not None:
            raise RuntimeError("Cannot set live points")

    @property
    def live_points(self) -> np.ndarray:
        return self.model.from_unit_hypercube(self.live_points_unit)

    @live_points.setter
    def live_points(self, samples) -> None:
        if samples is not None:
            raise RuntimeError("Cannot set live points")

    @property
    def nested_samples_unit(self) -> np.ndarray:
        """The current set of discarded points"""
        return self._ordered_samples.nested_samples

    @property
    def nested_samples(self) -> np.ndarray:
        return self.model.from_unit_hypercube(self.nested_samples_unit)

    @property
    def state(self) -> _INSIntegralState:
        return self._ordered_samples.state

    @property
    def final_samples_unit(self) -> np.ndarray:
        if self._final_samples is not None:
            return self._final_samples.samples
        elif self.iid_samples is not None:
            return self.iid_samples.samples
        else:
            return None

    @property
    def final_samples(self) -> np.ndarray:
        return self.model.from_unit_hypercube(self.final_samples_unit)

    @property
    def final_state(self) -> _INSIntegralState:
        if self._final_samples is not None:
            return self._final_samples.state
        elif self.iid_samples is not None:
            return self.iid_samples.state
        else:
            return None

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
        add_extra_parameters_to_live_points(["logW", "logQ"])

    def configure_stopping_criterion(
        self,
        stopping_criterion: Union[str, List[str]],
        tolerance: Union[float, List[float]],
        check_criteria: Literal["any", "all"],
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
                f"Unknown stopping criterion: {stopping_criterion}"
            )
        for c, c_use in zip(stopping_criterion, self.stopping_criterion):
            if c != c_use:
                logger.info(
                    f"Stopping criterion specified ({c}) is "
                    f"an alias for {c_use}. Using {c_use}."
                )
        if len(self.stopping_criterion) != len(self.tolerance):
            raise ValueError(
                "Number of stopping criteria must match tolerances"
            )
        self.criterion = len(self.tolerance) * [np.inf]

        logger.info(f"Stopping criteria: {self.stopping_criterion}")
        logger.info(f"Tolerance: {self.tolerance}")

        if check_criteria not in {"any", "all"}:
            raise ValueError("check_criteria must be any or all")
        if check_criteria == "any":
            self._stop_any = True
        else:
            self._stop_any = False

    def get_proposal(self, subdir: str = "levels", **kwargs):
        """Configure the proposal."""
        output = os.path.join(self.output, subdir, "")
        proposal = ImportanceFlowProposal(
            self.model, output, self.n_initial, **kwargs
        )
        return proposal

    def configure_iterations(
        self,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
    ) -> None:
        """Configure the minimum and maximum iterations.

        Note: will override any existing values when called.
        """
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
            raise ValueError("`min_samples` must be less than `nlive`")
        if self.min_remove > self.nlive:
            raise ValueError("`min_remove` must be less than `nlive`")
        logger.debug("Sampler configuration is valid")
        return True

    def populate_live_points(self) -> None:
        """Draw the initial live points from the prior.

        The live points are automatically sorted and assigned the iteration
        number -1.
        """
        if self.draw_iid_live:
            target = int(2 * self.n_initial)
        else:
            target = self.n_initial
        live_points = np.empty(target, dtype=get_dtype(self.model.names))
        n = 0
        logger.debug(f"Drawing {self.n_initial} initial points")
        while n < target:
            points = self.model.sample_unit_hypercube(target)
            points["logP"] = self.model.batch_evaluate_log_prior(
                points, unit_hypercube=True
            )
            accept = np.isfinite(points["logP"])
            n_it = accept.sum()
            m = min(n_it, target - n)
            live_points[n : (n + m)] = points[accept][:m]
            n += m

        live_points["logL"] = self.model.batch_evaluate_log_likelihood(
            live_points, unit_hypercube=True
        )
        if not np.isfinite(live_points["logL"]).all():
            logger.warning("Found infinite values in the log-likelihood")

        if np.any(live_points["logL"] == np.inf):
            raise RuntimeError("Live points contain +inf log-likelihoods")

        live_points["it"] = -np.ones(live_points.size)
        # Since log_Q is computed in the unit-cube
        live_points["logQ"] = np.zeros(live_points.size)
        live_points["logW"] = -live_points["logQ"]
        log_q = np.zeros([live_points.size, 1])

        if self.draw_iid_live:
            live_points, iid_samples = (
                live_points[: self.n_initial],
                live_points[self.n_initial :],
            )
            log_q, iid_log_q = (
                log_q[: self.n_initial, ...],
                log_q[self.n_initial :, ...],
            )
            self.iid_samples.add_initial_samples(iid_samples, iid_log_q)

        self.training_samples.add_initial_samples(live_points, log_q)

    def initialise(self) -> None:
        """Initialise the nested sampler.

        Draws live points, initialises the proposal.
        """
        if self.initialised:
            logger.warning("Nested sampler has already been initialised!")
        if self.live_points_unit is None:
            self.populate_live_points()

        self.initialise_history()
        self.proposal.initialise()
        self.initialised = True

    def initialise_history(self) -> None:
        """Initialise the dictionary to store history"""
        if self.history is None:
            super().initialise_history()
            self.history.update(
                dict(
                    min_log_likelihood=[],
                    max_log_likelihood=[],
                    logL_threshold=[],
                    logX=[],
                    gradients=[],
                    median_logL=[],
                    leakage_live_points=[],
                    leakage_new_points=[],
                    logZ=[],
                    n_live=[],
                    n_added=[],
                    n_removed=[],
                    n_post=[],
                    live_points_ess=[],
                    pool_entropy=[],
                    samples_entropy=[],
                    proposal_entropy=[],
                    stopping_criteria={
                        k: [] for k in self.stopping_criterion_aliases.keys()
                    },
                )
            )
        else:
            logger.debug("History dictionary already initialised")

    def update_history(self) -> None:
        """Update the history dictionary"""
        self.history["min_log_likelihood"].append(
            np.min(self.live_points_unit["logL"])
        )
        self.history["max_log_likelihood"].append(
            np.max(self.live_points_unit["logL"])
        )
        self.history["median_logL"].append(
            np.median(self.live_points_unit["logL"])
        )
        self.history["logL_threshold"].append(self.log_likelihood_threshold)
        self.history["logX"].append(self.logX)
        self.history["gradients"].append(self.gradient)
        self.history["logZ"].append(self.state.logZ)
        self.history["n_post"].append(self.state.effective_n_posterior_samples)
        self.history["samples_entropy"].append(self.samples_entropy)
        self.history["proposal_entropy"].append(self.current_proposal_entropy)
        self.history["live_points_ess"].append(self.live_points_ess)
        self.history["likelihood_evaluations"].append(
            self.model.likelihood_evaluations
        )

        for k in self.stopping_criterion_aliases.keys():
            self.history["stopping_criteria"][k].append(
                getattr(self, k, np.nan)
            )

    def determine_threshold_quantile(
        self,
        samples: np.ndarray,
        q: float = 0.8,
        include_likelihood: bool = False,
    ) -> int:
        """Determine where the next likelihood threshold should be located.

        Computes the q'th quantile based on log-likelihood and log-weights.

        Parameters
        ----------
        samples : np.ndarray
            An array of samples
        q : float
            Quantile to use. Defaults to 0.8
        include_likelihood : bool
            If True, the likelihood is included in the weights.

        Returns
        -------
        int
            The number of live points to discard.
        """
        logger.debug(f"Determining {q:.3f} quantile")
        a = samples["logL"]
        if include_likelihood:
            log_weights = samples["logW"] + samples["logL"]
        else:
            log_weights = samples["logW"].copy()
        cutoff = weighted_quantile(
            a, q, log_weights=log_weights, values_sorted=True
        )
        if not np.isfinite(cutoff):
            raise RuntimeError("Could not determine valid quantile")
        n = np.argmax(a >= cutoff)
        logger.debug(f"{q:.3} quantile is logL ={cutoff}")
        return int(n)

    def determine_threshold_entropy(
        self,
        samples: np.ndarray,
        q: float = 0.5,
        include_likelihood: bool = False,
        use_log_weights: bool = True,
    ) -> int:
        """Determine where the next likelihood threshold should be located
        using the entropy method.

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
            log_weights = samples["logW"] + samples["logL"]
        else:
            log_weights = samples["logW"]
        if use_log_weights:
            p = log_weights
        else:
            p = np.exp(log_weights)
        cdf = np.cumsum(p)
        if cdf.sum() == 0:
            cdf = np.arange(len(p), dtype=float)
        cdf /= cdf[-1]
        n = np.argmax(cdf >= q)
        if self.plot and self._plot_level_cdf:
            output = os.path.join(
                self.output, "levels", f"level_{self.iteration}"
            )
            os.makedirs(output, exist_ok=True)
            self.plot_level_cdf(
                samples["logL"],
                cdf,
                threshold=samples["logL"][n],
                q=q,
                filename=os.path.join(output, "cdf.png"),
            )
        return int(n)

    @nessai_style
    def plot_level_cdf(
        self,
        log_likelihood_values: np.ndarray,
        cdf: np.ndarray,
        threshold: float,
        q: float,
        filename: Optional[str] = None,
    ) -> Union[matplotlib.figure.Figure, None]:
        """Plot the CDF of the log-likelihood

        Parameters
        ----------
        log_likelihood_values : np.ndarray
            The log-likelihood values for the CDF
        cdf : np.ndarray
            The CDF to plot
        filename : Optional[str]
            Filename for saving the figure. If not specified the figure will
            be returned instead.

        Returns
        -------
        matplotlib.figure.Figure
            Level CDF figure. Only returned when the filename is not
            specified.
        """
        fig = plt.figure()
        plt.plot(log_likelihood_values, cdf)
        plt.xlabel("Log-likelihood")
        plt.title("CDF")
        plt.axhline(q, c="C1")
        plt.axvline(threshold, c="C1")

        if filename is not None:
            fig.savefig(filename)
            plt.close()
        else:
            return fig

    def determine_log_likelihood_threshold(
        self,
        samples: np.ndarray,
        method: Literal["entropy", "quantile"] = "entropy",
        **kwargs,
    ) -> int:
        """Determine the next likelihood threshold

        Parameters
        ----------
        samples : numpy.ndarray
            An array of samples
        method : Literal["entropy", "quantile"]
            The method to use
        kwargs :
            Keyword arguments passed to the function for the chosen method.


        Returns
        -------
        float
            The log-likelihood threshold
        """
        if method == "quantile":
            n = self.determine_threshold_quantile(samples, **kwargs)
        elif method == "entropy":
            n = self.determine_threshold_entropy(samples, **kwargs)
        else:
            raise ValueError(method)
        logger.debug(f"Next iteration should remove {n} points")
        if n == 0:
            if self.min_remove < 1:
                return 0
            else:
                n = 1
        if (samples.size - n) < self.min_samples:
            logger.warning(
                f"Cannot remove {n} from {samples.size}, "
                f"min_samples={self.min_samples}"
            )
            n = max(0, samples.size - self.min_samples)
        elif n < self.min_remove:
            logger.warning(
                f"Cannot remove less than {self.min_remove} samples"
            )
            n = self.min_remove

        if (
            self.draw_constant
            and self.max_samples
            and ((samples.size - n) + self.nlive) > self.max_samples
        ):
            n = samples.size - self.max_samples + self.nlive
            logger.warning(
                "Next level would have more than max samples, "
                f"removing {n} samples"
            )

        threshold = samples[n]["logL"].copy()
        return threshold

    def update_log_likelihood_threshold(self, threshold: float) -> None:
        self.log_likelihood_threshold = threshold
        self.training_samples.update_log_likelihood_threshold(
            self.log_likelihood_threshold
        )
        if self.iid_samples:
            self.iid_samples.update_log_likelihood_threshold(
                self.log_likelihood_threshold
            )

    def add_new_proposal(self):
        """Add a new proposal to the meta proposal"""
        st = datetime.datetime.now()

        # Implicitly includes all samples
        # Ensure at least min_samples are used for training
        # This reduces the likelihood threshold but avoids issues with training
        # with e.g. 1 point.
        n_train = min(
            np.argmax(
                self.training_samples.samples["logL"]
                >= self.log_likelihood_threshold
            ),
            self.training_samples.samples.size - self.min_samples,
        )
        self.current_training_samples = self.training_samples.samples[
            n_train:
        ].copy()
        self.current_training_log_q = self.training_samples.log_q[
            n_train:, :
        ].copy()

        logger.info(
            "Training next proposal with "
            f"{len(self.current_training_samples)} samples"
        )

        logger.debug("Updating the contour")
        logger.debug(
            "Training data ESS: "
            f"{effective_sample_size(self.current_training_samples['logW'])}"
        )

        if self.replace_all:
            weights = -np.exp(self.current_training_log_q[:, -1])
        elif self.weighted_kl:
            log_w = self.current_training_samples["logW"].copy()
            log_w -= logsumexp(log_w)
            weights = np.exp(log_w)
        else:
            weights = None

        self.proposal.train(
            self.current_training_samples,
            plot=self.plot_training_data,
            weights=weights,
        )
        self.training_time += datetime.datetime.now() - st

    def draw_n_samples(self, n: int, **kwargs):
        """Draw n samples from the current proposal

        Includes computing the log-likelihood of the samples

        Parameters
        ----------
        n : int
            Number of samples to draw
        kwargs :
            Keyword arguments passed to the :code:`draw` method of the proposal
            class.

        """
        st = datetime.datetime.now()
        logger.debug(f"Drawing {n} samples from the proposal")
        new_points, log_q = self.proposal.draw(n, **kwargs)
        logger.debug("Evaluating likelihood for new points")
        new_points["logL"] = self.model.batch_evaluate_log_likelihood(
            new_points,
            unit_hypercube=True,
        )
        logger.debug(
            "Min. log-likelihood of new samples: "
            f"{np.min(new_points['logL'])}"
        )
        if not np.isfinite(new_points["logL"]).all():
            logger.warning("Log-likelihood contains infs")

        if np.any(new_points["logL"] == -np.inf):
            logger.warning("New points contain points with zero likelihood")

        self.draw_samples_time += datetime.datetime.now() - st
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
                np.sum(
                    samples["logW"][
                        samples["logL"] < self.log_likelihood_threshold
                    ]
                )
                / samples["logW"].sum()
            )
        else:
            return (
                samples["logL"] < self.log_likelihood_threshold
            ).sum() / samples.size

    def add_and_update_points(self, n: int):
        """Add new points to the current set of live points.

        Parameters
        ----------
        n : int
            The number of points to add.
        """
        st = datetime.datetime.now()
        logger.info(f"Drawing {n} new samples from the new proposal")
        new_samples, log_q = self.draw_n_samples(n, update_counts=True)
        new_samples["it"] = self.iteration

        self.history["leakage_new_points"].append(
            self.compute_leakage(new_samples)
        )

        self._current_proposal_entropy = differential_entropy(-log_q[:, -1])

        logger.debug(
            "New samples ESS: " f"{effective_sample_size(new_samples['logW'])}"
        )

        if self.plot and self.plot_pool:
            plot_1d_comparison(
                self.current_training_samples,
                new_samples,
                filename=os.path.join(
                    self.output, "levels", f"pool_{self.iteration}.png"
                ),
            )

        self.training_samples.log_q = self.proposal.update_log_q(
            self.training_samples.samples, self.training_samples.log_q
        )

        self.training_samples.samples["logQ"] = (
            self.proposal.compute_meta_proposal_from_log_q(
                self.training_samples.log_q
            )
        )
        self.training_samples.samples["logW"] = -self.training_samples.samples[
            "logQ"
        ]

        self.training_samples.add_samples(new_samples, log_q)

        if self.draw_iid_live:
            iid_samples, iid_log_q = self.draw_n_samples(
                n, update_counts=False
            )
            iid_samples["it"] = self.iteration

            self.iid_samples.log_q = self.proposal.update_log_q(
                self.iid_samples.samples, self.iid_samples.log_q
            )
            self.iid_samples.samples["logQ"] = (
                self.proposal.compute_meta_proposal_from_log_q(
                    self.iid_samples.log_q
                )
            )
            self.iid_samples.samples["logW"] = -self.iid_samples.samples[
                "logQ"
            ]
            self.iid_samples.add_samples(iid_samples, iid_log_q)

        self.history["n_added"].append(new_samples.size)

        self.history["n_live"].append(self.live_points_unit.size)
        self.live_points_ess = effective_sample_size(
            self.live_points_unit["logW"]
        )
        self.history["leakage_live_points"].append(
            self.compute_leakage(self.live_points_unit)
        )
        logger.info(f"Current n samples: {self.live_points_unit.size}")
        logger.debug(f"Current live points ESS: {self.live_points_ess:.2f}")

        self.add_and_update_samples_time += datetime.datetime.now() - st

    def remove_samples(self) -> int:
        """Remove samples from the current set of live points."""
        n_removed = self.training_samples.remove_samples()
        if self.draw_iid_live:
            n_removed = self.iid_samples.remove_samples()
        self.history["n_removed"].append(n_removed)
        return n_removed

    def adjust_final_samples(self, n_batches=5):
        """Adjust the final samples"""
        orig_n_total = len(self.samples_unit)
        its, counts = np.unique(self.samples_unit["it"], return_counts=True)
        assert counts.sum() == orig_n_total
        weights = counts / orig_n_total
        original_unnorm_weight = counts.copy()
        norm_weight = original_unnorm_weight / original_unnorm_weight.sum()

        logger.debug(f"Final counts: {counts}")
        logger.debug(f"Final weights: {weights}")
        logger.debug(f"Final its: {list(self.proposal.n_requested.keys())}")

        sort_idx = np.argsort(self.samples_unit, order="it")
        samples = self.samples_unit.copy()[sort_idx]
        log_q = self.log_q[sort_idx]
        n_total = len(samples)

        # This changes the proposal because the number of samples changes
        log_evidences = np.empty(n_batches)
        log_evidence_errors = np.empty(n_batches)
        proposal = self.proposal
        for i in range(n_batches):
            new_counts = np.random.multinomial(
                orig_n_total,
                weights,
            )
            logger.debug(f"New counts: {new_counts}")
            logger.debug(new_counts.sum())

            # Draw missing samples
            for it, c, nc in zip(its, counts, new_counts):
                if nc > c:
                    logger.debug(f"Drawing {nc - c} samples from {it}")
                    if it == -1:
                        new_samples, new_log_q = proposal.draw_from_prior(
                            nc - c
                        )
                    else:
                        new_samples, new_log_q = proposal.draw(
                            n=(nc - c),
                            flow_number=it,
                            update_counts=False,
                        )
                    new_samples["it"] = it
                    new_samples["logL"] = (
                        self.model.batch_evaluate_log_likelihood(
                            new_samples, unit_hypercube=True
                        )
                    )
                    new_loc = np.searchsorted(samples["it"], new_samples["it"])
                    samples = np.insert(samples, new_loc, new_samples)
                    log_q = np.insert(log_q, new_loc, new_log_q, axis=0)
                    n_total = samples.size
                    counts = np.unique(samples["it"], return_counts=True)[1]
                    logger.debug(f"Updated counts: {counts}")

            idx_keep = np.zeros(n_total, dtype=bool)
            cc = 0
            for it, c, nc in zip(its, counts, new_counts):
                assert c >= nc
                idx = np.random.choice(
                    np.arange(cc, cc + c), size=nc, replace=False
                )
                idx_keep[idx] = True
                assert np.all(samples[idx]["it"] == it)
                cc += c

            batch_samples = samples[idx_keep]
            batch_log_q = log_q[idx_keep]
            assert batch_samples.size == orig_n_total

            log_Q = logsumexp(batch_log_q, b=norm_weight, axis=1)
            # Weights are normalised because the total number of samples is the
            # same.
            batch_samples["logQ"] = log_Q
            batch_samples["logW"] = -log_Q
            state = _INSIntegralState()
            state.log_meta_constant = 0.0
            state.update_evidence(batch_samples)
            log_evidences[i] = state.log_evidence
            log_evidence_errors[i] = state.log_evidence_error
            logger.debug(f"Log-evidence batch {i} = {log_evidences[i]:.3f}")

        mean_log_evidence = np.mean(log_evidences)
        standard_error = np.std(log_evidences, ddof=1)

        logger.info(f"Mean log evidence: {mean_log_evidence:.3f}")
        logger.info(f"SE log evidence: {standard_error:.3f}")
        self.bootstrap_log_evidence = mean_log_evidence
        self.bootstrap_log_evidence_error = standard_error

    def finalise(self) -> None:
        """Finalise the sampling process."""
        if self.finalised:
            logger.warning("Sampler already finalised")
            return
        logger.info("Finalising")

        if self._train_final_flow:
            self.train_final_flow()

        self.training_samples.finalise()
        if self.draw_iid_live:
            self.iid_samples.finalise()

        if self.bootstrap:
            self.adjust_final_samples()

        final_kl = self.kl_divergence(self.samples_unit)

        logger.info(
            f"Final log Z: {self.state.logZ:.3f} "
            f"+/- {self.state.compute_uncertainty():.3f}"
        )
        logger.info(f"Final KL divergence: {final_kl:.3f}")
        logger.info(
            f"Final ESS: {self.state.effective_n_posterior_samples:.3f}"
        )
        self.finalised = True
        self.checkpoint(periodic=True, force=True)
        self.produce_plots()

    def add_level_post_sampling(self, samples: np.ndarray, n: int) -> None:
        """Add a level to the nested sampler after initial sampling has \
            completed.
        """
        self.proposal.train(samples)
        new_samples, log_q = self.draw_n_samples(n)
        log_q = self.update_live_points(new_samples, log_q)
        self.update_nested_samples()
        self.add_to_nested_samples(new_samples)
        self.state.update_evidence(self.nested_samples_unit)

    def compute_stopping_criterion(self) -> List[float]:
        """Compute the stopping criterion.

        The method used will depend on how the sampler was configured.
        """
        if self.iteration > 0:
            self.log_dZ = np.abs(self.log_evidence - self.history["logZ"][-1])
        else:
            self.log_dZ = np.inf
        self.ratio = self._ordered_samples.compute_evidence_ratio()
        self.ratio_ns = self.state.compute_evidence_ratio(ns_only=True)
        self.ess = self.state.effective_n_posterior_samples
        self.Z_err = np.exp(self.log_evidence_error)
        cond = [getattr(self, sc) for sc in self.stopping_criterion]

        logger.info(
            f"Stopping criteria ({self.stopping_criterion}): {cond} "
            f"- Tolerance: {self.tolerance}"
        )
        return cond

    def checkpoint(self, periodic: bool = False, force: bool = False):
        """Checkpoint the sampler."""
        if periodic is False:
            logger.warning(
                "Importance Sampler cannot checkpoint mid iteration"
            )
            return
        super().checkpoint(
            periodic=periodic,
            force=force,
            save_existing=self.save_existing_checkpoint,
        )

    def _compute_gradient(self) -> None:
        self.logX_pre = self.logX
        self.logX = logsumexp(self.live_points_unit["logW"])
        self.logL_pre = self.logL
        self.logL = logsumexp(
            self.live_points_unit["logL"] - self.live_points_unit["logQ"]
        )
        self.dlogX = self.logX - self.logX_pre
        self.dlogL = self.logL - self.logL_pre
        self.gradient = self.dlogL / self.dlogX

    def log_state(self):
        """Log the state of the sampler"""
        logger.info(
            f"Update {self.iteration} - "
            f"log Z: {self.state.logZ:.3f} +/- "
            f"{self.state.compute_uncertainty():.3f} "
            f"ESS: {self.ess:.1f} "
            f"logL min: {self.live_points_unit['logL'].min():.3f} "
            f"logL median: {np.nanmedian(self.live_points_unit['logL']):.3f} "
            f"logL max: {self.live_points_unit['logL'].max():.3f}"
        )

    def update_evidence(self):
        self.training_samples.update_evidence()
        if self.draw_iid_live:
            self.iid_samples.update_evidence()

    def compute_importance(self, **kwargs):
        if self.draw_iid_live:
            importance = self.iid_samples.compute_importance(**kwargs)
        else:
            importance = self.training_samples.compute_importance(**kwargs)
        return importance

    def nested_sampling_loop(self):
        """Main nested sampling loop."""
        if self.finalised:
            logger.warning("Sampler has already finished sampling! Aborting")
            return self.log_evidence, self.nested_samples_unit
        self.initialise()
        logger.info("Starting the nested sampling loop")

        while True:
            if self.reached_tolerance and self.iteration >= self.min_iteration:
                break

            self._compute_gradient()

            if self.n_update is None:
                threshold = self.determine_log_likelihood_threshold(
                    self.live_points_unit,
                    method=self.threshold_method,
                    **self.threshold_kwargs,
                )
            else:
                threshold = self.live_points_unit[self.n_update]["logL"].copy()

            logger.info(f"Log-likelihood threshold: {threshold}")
            self.update_log_likelihood_threshold(threshold)

            n_removed = self.remove_samples()

            self.add_new_proposal()

            if self.draw_constant or self.replace_all:
                n_add = self.nlive
            else:
                n_add = n_removed
            self.add_and_update_points(n_add)

            self.update_evidence()

            self.importance = self.compute_importance(importance_ratio=0.5)

            self.criterion = self.compute_stopping_criterion()

            self.log_state()

            self.update_history()
            self.iteration += 1
            if not self.iteration % self.plotting_frequency:
                self.produce_plots()
            if self.checkpointing:
                self.checkpoint(periodic=True)
            if self.iteration >= self.max_iteration:
                break

        logger.info(
            f"Finished nested sampling loop after {self.iteration} iterations "
            f"with {self.stopping_criterion} = {self.criterion}"
        )
        self.finalise()
        logger.info(f"Training time: {self.training_time}")
        logger.info(f"Draw samples time: {self.draw_samples_time}")
        logger.info(
            f"Add and update samples time: {self.add_and_update_samples_time}"
        )
        logger.info(f"Log-likelihood time: {self.likelihood_evaluation_time}")
        return self.log_evidence, self.samples

    def draw_posterior_samples(
        self,
        sampling_method: str = "rejection_sampling",
        n: Optional[int] = None,
        use_final_samples: bool = True,
    ) -> np.ndarray:
        """Draw posterior samples from the current nested samples."""

        if use_final_samples and self.final_samples_unit is not None:
            samples = self.final_samples
            log_w = self.final_state.log_posterior_weights
        else:
            samples = self.samples
            log_w = self.state.log_posterior_weights

        posterior_samples, indices = draw_posterior_samples(
            samples,
            log_w=log_w,
            method=sampling_method,
            n=n,
            return_indices=True,
        )

        # TODO: check this is correct
        log_p = log_w[indices] - log_w[indices].max()
        h = differential_entropy(log_p)
        logger.debug(f"Information in the posterior: {h:.3f} nats")

        logger.info(f"Produced {posterior_samples.size} posterior samples.")
        return posterior_samples

    @staticmethod
    def kl_divergence(samples: np.ndarray) -> float:
        """Compute the KL divergence between the meta-proposal and posterior.

        Uses all samples drawn from the meta-proposal
        """
        if not len(samples):
            return np.inf
        # logQ is computed on the unit hyper-cube where the prior is 1/1^n
        # so logP = 0
        return np.mean(
            2 * samples["logQ"]
            + samples["logP"]
            + np.log(samples.size)
            - samples["logL"]
        )

    def draw_more_nested_samples(self, n: int) -> np.ndarray:
        """Draw more nested samples from g"""
        samples = self.proposal.draw_from_flows(n)
        samples["logL"] = self.model.batch_evaluate_log_likelihood(
            samples, unit_hypercube=True
        )
        state = _INSIntegralState()
        state.update_evidence(samples)
        logger.info(
            "Evidence in new nested samples: "
            f"{state.logZ:3f} +/- {state.compute_uncertainty():.3f}"
        )
        logger.info(
            "Effective number of posterior samples: "
            f"{state.effective_n_posterior_samples:3f}"
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
        optimisation_method: Literal["evidence", "kl"] = "kl",
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
        logger.info("Drawing final samples")
        if n_post and n_draw:
            raise RuntimeError("Specify either `n_post` or `n_draw`")
        start_time = datetime.datetime.now()

        if self._final_samples:
            logger.warning("Existing final samples will be overridden")
        elif self.iid_samples:
            logger.warning("Already have i.i.d samples")

        final_samples = OrderedSamples()

        eff = (
            self.state.effective_n_posterior_samples
            / self.nested_samples_unit.size
        )
        max_samples = int(max_samples_ratio * self.nested_samples_unit.size)

        max_log_likelihood = np.max(self.samples_unit["logL"])

        logger.debug(f"Expected efficiency: {eff:.3f}")
        if not any([n_post, n_draw]):
            n_draw = self.samples_unit.size

        if n_post:
            n_draw = int(n_post / eff)
            logger.info(f"Redrawing samples with target ESS: {n_post:.1f}")
            logger.info(f"Expect to draw approximately {n_draw:.0f} samples")
            if n_draw > max_samples:
                logger.warning(
                    f"Expected number of samples ({n_draw}) is greater than "
                    f"the maximum number of samples ({max_samples}). Final "
                    "ESS will most likely be less than the specified value."
                )
        else:
            logger.info(f"Drawing at least {n_draw} final samples")

        batch_size = int(1.05 * n_draw)
        while batch_size > max_batch_size:
            if batch_size <= 1:
                raise RuntimeError(
                    "Could not determine a valid batch size. "
                    "Consider changing the maximum batch size."
                )
            batch_size //= 2

        logger.debug(f"Batch size: {batch_size}")

        if optimise_weights:
            weights = self.imp_post
            if optimisation_method == "evidence":
                pass
            elif optimisation_method == "kl":
                if optimise_kwargs is None:
                    optimise_kwargs = {}
                weights = optimise_meta_proposal_weights(
                    self.nested_samples_unit,
                    self._log_q_ns,
                    initial_weights=weights,
                    **optimise_kwargs,
                )
            else:
                raise ValueError(optimisation_method)
            target_counts = None
        elif use_counts:
            logger.warning("Using counts is not recommended!")
            target_counts = np.array(
                np.fromiter(self.proposal.unnormalised_weights.values(), int)
                * (batch_size / self.proposal.normalisation_constant),
                dtype=int,
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

        it = 0
        ess = 0

        while True:
            if n_post and (ess > n_post):
                break
            if it >= max_its:
                logger.warning("Reached maximum number of iterations.")
                logger.warning("Stopping drawing final samples.")
                break
            if n_post is None and (samples.size > n_draw):
                break
            if max_samples_ratio and (len(samples) > max_samples):
                logger.warning(
                    f"Reached maximum number of samples: {max_samples}"
                )
                logger.warning("Stopping")
                break

            it_samples = np.empty([0], dtype=self.proposal.dtype)
            # Target counts will be None if use_counts is False
            it_samples, new_log_q, new_counts = self.proposal.draw_from_flows(
                batch_size,
                counts=target_counts,
                weights=weights,
            )
            log_q = np.concatenate([log_q, new_log_q], axis=0)
            counts += new_counts

            it_samples["logL"] = self.model.batch_evaluate_log_likelihood(
                it_samples,
                unit_hypercube=True,
            )

            if np.any(it_samples["logL"] > max_log_likelihood):
                logger.warning(
                    f"Max logL increased from {max_log_likelihood:.3f} to "
                    f"{it_samples['logL'].max():.3f}"
                )

            samples = np.concatenate([samples, it_samples])

            log_Q = logsumexp(log_q, b=weights, axis=1)

            if np.isposinf(log_Q).any():
                logger.warning("Log meta proposal contains +inf")

            samples["logQ"] = log_Q
            samples["logW"] = -log_Q

            final_samples.state.update_evidence(samples)
            ess = final_samples.state.effective_n_posterior_samples
            logger.debug(f"Sample count: {samples.size}")
            logger.debug(f"Current ESS: {ess}")
            it += 1
            logger.info(f"Drawn {samples.size} - ESS: {ess:2f}")

        logger.debug(f"Original weights: {self.proposal.unnormalised_weights}")
        logger.debug(f"New weights: {counts}")

        logger.info(f"Drew {samples.size} final samples")
        logger.info(
            f"Final log-evidence: {final_samples.state.log_evidence:.3f} "
            f"+/- {final_samples.state.log_evidence_error:.3f}"
        )
        logger.info(f"Final ESS: {ess:.1f}")
        final_samples.samples = samples
        final_samples.log_q = log_q
        self._final_samples = final_samples
        self.draw_final_samples_time += datetime.datetime.now() - start_time
        return self.final_state.logZ, samples

    def train_final_flow(self):
        """Train a final flow using all of the nested samples"""
        logger.warning("Training final flow")
        from ..flowmodel import FlowModel

        weights = np.exp(self.state.log_posterior_weights)
        weights /= weights.sum()
        samples, log_j = self.proposal.rescale(self.nested_samples_unit)
        flow = FlowModel(
            output=os.path.join(self.output, "levels", "final_level", ""),
            config=self.proposal.flow_config,
        )
        flow.initialise()
        flow.train(samples, weights=weights)

        x_p_out, log_prob = flow.sample_and_log_prob(
            self.nested_samples_unit.size
        )
        x_out, log_j_out = self.proposal.inverse_rescale(x_p_out)
        x_out["logQ"] = log_prob - log_j_out
        x_out["logW"] = -x_out["logQ"]
        x_out["logL"] = self.model.batch_evaluate_log_likelihood(
            x_out, unit_hypercube=True
        )

        state = _INSIntegralState(normalised=False)
        state.log_meta_constant = 0.0
        state.update_evidence(x_out)

    @nessai_style()
    def plot_state(
        self, filename: Optional[str] = None
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
        n_subplots = 8

        fig, ax = plt.subplots(n_subplots, 1, sharex=True, figsize=(15, 15))
        ax = ax.ravel()
        its = np.arange(self.iteration)

        for a in ax:
            a.vlines(self.history["checkpoint_iterations"], 0, 1, color="C2")

        # Counter for each plot
        m = 0

        ax[m].plot(
            its,
            self.history["min_log_likelihood"],
            label="Min. Log L",
        )
        ax[m].plot(
            its,
            self.history["max_log_likelihood"],
            label="Max. Log L",
        )
        ax[m].plot(
            its,
            self.history["median_logL"],
            label="Median Log L",
        )
        ax[m].set_ylabel("Log-likelihood")
        ax[m].legend()

        m += 1

        ax[m].plot(
            its,
            self.history["logL_threshold"],
        )
        ax[m].set_ylabel(r"$\log L_t$")
        m += 1

        ax[m].plot(
            its,
            self.history["logZ"],
            label="Log Z",
        )
        ax[m].set_ylabel("Log-evidence")
        ax[m].legend()

        ax_dz = plt.twinx(ax[m])
        ax_dz.plot(
            its,
            self.history["stopping_criteria"]["log_dZ"],
            label="log dZ",
            c="C1",
            ls=config.plotting.line_styles[1],
        )
        ax_dz.set_ylabel("log dZ")
        ax_dz.set_yscale("log")
        handles, labels = ax[m].get_legend_handles_labels()
        handles_dz, labels_dz = ax_dz.get_legend_handles_labels()
        ax[m].legend(handles + handles_dz, labels + labels_dz)

        m += 1

        ax[m].plot(its, self.history["likelihood_evaluations"])
        ax[m].set_ylabel("# likelihood \n evaluations")

        m += 1

        ax[m].plot(
            its,
            self.history["n_post"],
            label="Posterior",
        )
        ax[m].plot(
            its,
            self.history["live_points_ess"],
            label="Live points",
        )
        ax[m].set_ylabel("ESS")
        ax[m].legend()

        m += 1

        ax[m].plot(its, self.importance["total"][1:], label="Total")
        ax[m].plot(its, self.importance["posterior"][1:], label="Posterior")
        ax[m].plot(its, self.importance["evidence"][1:], label="Evidence")
        ax[m].legend()
        ax[m].set_ylabel("Level importance")

        m += 1

        ax[m].plot(
            its,
            self.history["n_removed"],
            label="Removed",
        )
        ax[m].plot(its, self.history["n_added"], label="Added")
        ax[m].plot(its, self.history["n_live"], label="Total")
        ax[m].set_ylabel("# samples")
        ax[m].legend()

        ax[m].legend()
        m += 1

        for (i, sc), tol in zip(
            enumerate(self.stopping_criterion), self.tolerance
        ):
            ax[m].plot(
                its,
                self.history["stopping_criteria"][sc],
                label=sc,
                c=f"C{i}",
                ls=config.plotting.line_styles[i],
            )
            ax[m].axhline(tol, ls=":", c=f"C{i}")
        ax[m].legend()
        ax[m].set_ylabel("Stopping criterion")

        ax[-1].set_xlabel("Iteration")

        fig.suptitle(
            f"Sampling time: {self.current_sampling_time}", fontsize=16
        )

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    @nessai_style
    def plot_extra_state(
        self,
        filename: Optional[str] = None,
    ) -> Union[matplotlib.figure.Figure, None]:
        """Produce a state plot that contains extra tracked statistics.

        Parameters
        ----------
        filename : Optional[str]
            Filename name for the plot when saved. If specified the figure will
            be saved, otherwise the figure is returned.

        Returns
        -------
        Union[matplotlib.figure.Figure, None]
            Returns the figure if a filename name is not given.
        """
        n_subplots = 4

        fig, ax = plt.subplots(n_subplots, 1, sharex=True, figsize=(15, 15))
        ax = ax.ravel()
        its = np.arange(self.iteration)

        for a in ax:
            a.vlines(self.checkpoint_iterations, 0, 1, color="C2")

        # Counter for each plot
        m = 0

        ax[m].plot(its, self.history["logX"])
        ax[m].set_ylabel("Log X")

        m += 1

        ax[m].plot(its, self.history["gradients"])
        ax[m].set_ylabel("dlogL/dlogX")

        m += 1

        ax[m].plot(
            its,
            self.history["leakage_live_points"],
            label="Total leakage",
        )
        ax[m].plot(
            its,
            self.history["leakage_new_points"],
            label="New leakage",
        )
        ax[m].set_ylabel("Leakage")
        ax[m].legend()

        m += 1

        ax[m].plot(
            its,
            self.history["samples_entropy"],
            label="Overall",
        )
        ax[m].plot(
            its,
            self.history["proposal_entropy"],
            label="Current",
        )
        ax[m].set_ylabel("Differential\n entropy")
        ax[m].legend()

        m += 1

        ax[-1].set_xlabel("Iteration")

        fig.suptitle(
            f"Sampling time: {self.current_sampling_time}", fontsize=16
        )

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    @nessai_style()
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

        parameters = list(self.samples_unit.dtype.names)
        for p in ["logW"]:
            parameters.remove(p)
        n = len(parameters)

        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(5, 2 * n))

        if enable_colours:
            colour_kwargs = dict(
                c=self.samples_unit["it"],
                vmin=-1,
                vmax=self.samples_unit["it"].max(),
            )
        else:
            colour_kwargs = {}

        for ax, p in zip(axs, parameters):
            ax.scatter(
                self.samples_unit["logW"],
                self.samples_unit[p],
                s=1.0,
                **colour_kwargs,
            )
            ax.set_ylabel(p)
        axs[-1].set_xlabel("Log W")

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    @nessai_style(line_styles=False)
    def plot_likelihood_levels(
        self,
        filename: Optional[str] = None,
        cmap: str = "viridis",
        max_bins: int = 50,
    ) -> Optional[matplotlib.figure.Figure]:
        """Plot the distribution of the likelihood at each level.

        Parameters
        ----------
        filename
            Name of the file for saving the figure. If not specified, then
            the figure is returned.
        cmap
            Name of colourmap to use. Must be a valid colourmap in matplotlib.
        max_bins
            The maximum number of bins allowed.
        """
        its = np.unique(self.samples_unit["it"])
        colours = plt.get_cmap(cmap)(np.linspace(0, 1, len(its)))
        vmax = np.max(self.samples_unit["logL"])
        vmin = np.ma.masked_invalid(
            self.samples_unit["logL"][self.samples_unit["it"] == its[-1]]
        ).min()

        fig, axs = plt.subplots(1, 2)
        for it, c in zip(its, colours):
            data = self.samples_unit["logL"][self.samples_unit["it"] == it]
            data = data[np.isfinite(data)]
            if not len(data):
                continue
            bins = auto_bins(data, max_bins=max_bins)
            for ax in axs:
                ax.hist(
                    data,
                    bins,
                    histtype="step",
                    color=c,
                    density=True,
                )
                ax.set_xlabel("Log-likelihood")

        axs[0].set_ylabel("Density")
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
            logger.debug("Producing plots")
            self.plot_state(os.path.join(self.output, "state.png"))
            if self._plot_trace:
                self.plot_trace(
                    filename=os.path.join(self.output, "trace.png"),
                    **self.trace_plot_kwargs,
                )
            if self._plot_likelihood_levels:
                self.plot_likelihood_levels(
                    os.path.join(self.output, "likelihood_levels.png")
                )
            if self._plot_extra_state:
                self.plot_extra_state(
                    os.path.join(self.output, "state_extra.png")
                )
        else:
            logger.debug("Skipping plots")

    def get_result_dictionary(self):
        """Get a dictionary contain the main results from the sampler."""
        d = super().get_result_dictionary()
        d["history"] = self.history
        d["training_samples"] = self.model.from_unit_hypercube(
            self.training_samples.samples
        )
        d["training_log_evidence"] = self.training_samples.state.log_evidence
        d["training_log_evidence_error"] = (
            self.training_samples.state.log_evidence_error
        )
        d["training_log_posterior_weights"] = (
            self.training_samples.state.log_posterior_weights
        )
        # Will all be None if the final samples haven't been drawn
        d["bootstrap_log_evidence"] = self.bootstrap_log_evidence
        d["bootstrap_log_evidence_error"] = self.bootstrap_log_evidence_error
        if self.iid_samples:
            d["iid_log_evidence"] = self.iid_samples.state.log_evidence
            d["iid_log_evidence_error"] = (
                self.iid_samples.state.log_evidence_error
            )
        d["samples"] = self.final_samples
        d["log_posterior_weights"] = self.final_log_posterior_weights
        d["log_evidence"] = self.final_log_evidence
        d["log_evidence_error"] = self.final_log_evidence_error

        d["training_time"] = self.training_time.total_seconds()
        d["draw_samples_time"] = self.draw_samples_time.total_seconds()
        d["add_and_update_samples_time"] = (
            self.add_and_update_samples_time.total_seconds()
        )
        d["draw_final_samples_time"] = (
            self.draw_final_samples_time.total_seconds()
        )
        d["proposal_importance"] = self.importance

        return d

    @classmethod
    def resume_from_pickled_sampler(
        cls, sampler, model, flow_config=None, weights_path=None, **kwargs
    ):
        """Resume from a pickled sampler.

        Parameters
        ----------
        sampler : Any
            Pickled sampler.
        model : :obj:`nessai.model.Model`
            User-defined model
        flow_config : Optional[dict]
            Dictionary for configuring the flow
        weights_path : Optional[dict]
            Path to the weights files that will override the value stored in
            the proposal.
        kwargs :
            Keyword arguments passed to the parent class's method.

        Returns
        -------
        Instance of ImportanceNestedSampler
        """
        cls.add_fields()
        obj = super(ImportanceNestedSampler, cls).resume_from_pickled_sampler(
            sampler, model, **kwargs
        )
        logger.info(f"Resuming sampler at iteration {obj.iteration}")
        logger.info(f"Current number of samples: {len(obj.nested_samples)}")
        logger.info(
            f"Current log-evidence: {obj.log_evidence:3f} "
            f"+/- {obj.log_evidence_error:.3f}"
        )
        if flow_config is None:
            flow_config = {}
        obj.proposal.resume(model, flow_config, weights_path=weights_path)

        if obj.training_samples.log_q is None:
            logger.info("Recomputing log_q")
            (
                _,
                obj.training_samples.log_q,
            ) = obj.proposal.compute_meta_proposal_samples(
                obj.training_samples.samples
            )
        if obj.iid_samples and obj.iid_samples.log_q is None:
            logger.info("Recomputing log_q for i.i.d samples")
            (
                _,
                obj.iid_samples.log_q,
            ) = obj.proposal.compute_meta_proposal_samples(
                obj.iid_samples.samples
            )
        logger.info("Finished resuming sampler")
        return obj

    def __getstate__(self):
        d = self.__dict__
        exclude = {
            "model",
            "proposal",
            "checkpoint_callback",
            "training_samples",
            "iid_samples",
        }
        state = {k: d[k] for k in d.keys() - exclude}
        if d.get("model") is not None:
            state["_previous_likelihood_evaluations"] = d[
                "model"
            ].likelihood_evaluations
            state["_previous_likelihood_evaluation_time"] = d[
                "model"
            ].likelihood_evaluation_time.total_seconds()
        else:
            state["_previous_likelihood_evaluations"] = 0
            state["_previous_likelihood_evaluation_time"] = 0
        return state, self.proposal, self.training_samples, self.iid_samples

    def __setstate__(self, state):
        self.__dict__.update(state[0])
        self.proposal = state[1]
        self.training_samples = state[2]
        self.iid_samples = state[3]
