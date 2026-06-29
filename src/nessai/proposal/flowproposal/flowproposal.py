"""Proposal class that uses normalising flows and rejection sampling."""

from __future__ import annotations

import datetime
import logging
import math
import warnings

import numpy as np
from scipy.special import logsumexp

from ... import config
from ...livepoint import empty_structured_array
from ...utils.structures import get_subset_arrays
from .base import BaseFlowProposal
from .truncation import (
    TruncationScheme,
    apply_default_truncation_config,
    build_truncation_methods,
    get_deprecated_latent_radius_arguments,
    get_deprecated_latent_radius_kwargs,
    get_truncation_rule_class,
    normalise_truncation_kwargs,
)

logger = logging.getLogger(__name__)


def _clip_weights(weights: np.ndarray) -> np.ndarray:
    """Clip the largest ``ceil(sqrt(N))`` weights to their mean."""
    weights = weights.copy()
    num_clip = math.ceil(math.sqrt(len(weights)))
    if not num_clip:
        return weights
    if num_clip >= len(weights):
        clip_idx = np.arange(len(weights))
    else:
        clip_idx = np.argpartition(-weights, num_clip - 1)[:num_clip]
    weights[clip_idx] = weights[clip_idx].mean()
    weights /= weights.mean()
    return weights


class FlowProposal(BaseFlowProposal):
    """Proposal that samples in latent space using the trained flow.

    Parameters
    ----------
    model : nessai.model.Model
        The model to use for the proposal.
    poolsize : int, optional
        The number of samples to draw from the flow when populating the pool.
    latent_prior : str, optional
        The prior to use for the latent space. This argument is deprecated and
        only 'flow' is supported.
    latent_temperature : float, optional
        The temperature to use for the latent space. If None, no scaling is
        applied.
    constant_volume_mode : bool, optional
        Whether to use constant volume mode for the latent radius. This argument
        is deprecated and should be configured via :code:`truncation_methods`
        and :code:`truncation_kwargs` instead.
    volume_fraction : float, optional
        The volume fraction to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    fuzz : float, optional
        The fuzz to use for the latent radius. This argument is deprecated and
        should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    fixed_radius : float, optional
        The fixed radius to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    radius_mode : str, optional
        The radius mode to use for the latent radius. This argument is deprecated
        and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    drawsize : int, optional
        The number of samples to draw from the flow when populating the pool.
    truncate_log_q : bool, optional
        Whether to truncate the log q values when populating the pool. This
        argument is deprecated and should be configured via
        :code:`truncation_methods` and :code:`truncation_kwargs` instead.
    expansion_fraction : float, optional
        The expansion fraction to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    min_radius : float, optional
        The minimum radius to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    max_radius : float, optional
        The maximum radius to use for the latent radius. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    compute_radius_with_all : bool, optional
        Whether to compute the latent radius using all samples. This argument is
        deprecated and should be configured via :code:`truncation_methods` and
        :code:`truncation_kwargs` instead.
    enforce_likelihood_threshold : bool, optional
        Whether to enforce a likelihood threshold when populating the pool. This
        argument is deprecated and should be configured via
        :code:`truncation_methods` and :code:`truncation_kwargs` instead.
    truncation_method : str, optional
        The truncation method to use when populating the pool.
    truncation_methods : list of str, optional
        The truncation methods to use when populating the pool.
    truncation_kwargs : dict, optional
        The keyword arguments to use for the truncation methods when populating
        the pool. When using :code:`truncation_methods`, the keys of this
        dictionary should match the names of the truncation methods.
    **kwargs
        Additional keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        model,
        poolsize=None,
        latent_prior=None,
        latent_temperature=None,
        constant_volume_mode=None,
        volume_fraction=None,
        fuzz=None,
        fixed_radius=None,
        radius_mode=None,
        drawsize=None,
        truncate_log_q=False,
        expansion_fraction=None,
        min_radius=None,
        max_radius=None,
        compute_radius_with_all=None,
        enforce_likelihood_threshold=False,
        truncation_method=None,
        truncation_methods=None,
        truncation_kwargs=None,
        clip_population_weights=False,
        **kwargs,
    ):
        super().__init__(model, poolsize=poolsize, **kwargs)
        logger.debug("Initialising FlowProposal")
        self._warn_for_deprecated_latent_prior(latent_prior)

        self.configure_population(
            drawsize,
            latent_prior=latent_prior,
            latent_temperature=latent_temperature,
            clip_population_weights=clip_population_weights,
        )

        self._truncation_scheme = TruncationScheme()
        self.truncation_method = truncation_method
        self.truncation_methods = []
        self.truncation_kwargs = {}
        deprecated_latent_radius_arguments = dict(
            fixed_radius=fixed_radius,
            radius_mode=radius_mode,
            min_radius=min_radius,
            max_radius=max_radius,
            compute_radius_with_all=compute_radius_with_all,
            constant_volume_mode=constant_volume_mode,
            volume_fraction=volume_fraction,
            fuzz=fuzz,
            expansion_fraction=expansion_fraction,
        )
        self._warn_for_deprecated_truncation_arguments(
            **deprecated_latent_radius_arguments
        )
        self._warn_for_deprecated_likelihood_threshold(
            enforce_likelihood_threshold
        )
        latent_radius_kwargs = get_deprecated_latent_radius_kwargs(
            **deprecated_latent_radius_arguments
        )

        self.configure_truncation(
            truncation_method=truncation_method,
            truncation_methods=truncation_methods,
            truncation_kwargs=truncation_kwargs,
            truncate_log_q=truncate_log_q,
            enforce_likelihood_threshold=enforce_likelihood_threshold,
            latent_radius_kwargs=latent_radius_kwargs or None,
            default_latent_radius=True,
        )
        self._log_configuration()

    @property
    def truncation(self) -> TruncationScheme:
        return self._truncation_scheme

    def get_truncation_rule(self, name: str):
        return self.truncation.get_rule(name)

    def _get_latent_radius_rule(self):
        return self._truncation_scheme.get_rule("latent_radius")

    def _sync_truncation_state(self) -> None:
        self.truncation_methods = self._truncation_scheme.rule_names
        self.truncate_log_q = "min_log_q" in self.truncation_methods
        self.enforce_likelihood_threshold = (
            "likelihood_threshold" in self.truncation_methods
        )

    def _log_configuration(self) -> None:
        logger.info(
            "FlowProposal configuration: drawsize=%s, latent_prior=%s, "
            "latent_temperature=%s, truncation_methods=%s",
            self.drawsize,
            self.latent_prior,
            self.latent_temperature,
            self.truncation_methods,
        )
        for rule in self._truncation_scheme.rules:
            if hasattr(rule, "to_kwargs"):
                logger.info(
                    "FlowProposal truncation rule %s: %s",
                    rule.name,
                    rule.to_kwargs(),
                )
            else:
                logger.info(
                    "FlowProposal truncation rule enabled: %s",
                    rule.name,
                )

    def _warn_for_deprecated_truncation_arguments(self, **kwargs) -> None:
        deprecated = get_deprecated_latent_radius_arguments(**kwargs)
        if not deprecated:
            return

        warnings.warn(
            "The following FlowProposal arguments are deprecated and should "
            "be configured via truncation_kwargs['latent_radius'] instead: "
            f"{', '.join(deprecated)}.",
            FutureWarning,
            stacklevel=3,
        )

    @staticmethod
    def _warn_for_deprecated_latent_prior(latent_prior) -> None:
        if latent_prior != "flow":
            return
        warnings.warn(
            "`latent_prior` is deprecated and will be removed. "
            "FlowProposal always uses the flow latent distribution.",
            FutureWarning,
            stacklevel=3,
        )

    @staticmethod
    def _warn_for_deprecated_likelihood_threshold(
        enforce_likelihood_threshold: bool,
    ) -> None:
        if not enforce_likelihood_threshold:
            return
        warnings.warn(
            "`enforce_likelihood_threshold` is deprecated and should be "
            "configured via "
            "`truncation_methods=['likelihood_threshold']` instead.",
            FutureWarning,
            stacklevel=3,
        )

    def configure_population(
        self,
        drawsize,
        latent_prior=None,
        latent_temperature=None,
        clip_population_weights=False,
    ) -> None:
        """Configure settings related to population."""
        if drawsize is None:
            drawsize = self.poolsize

        if latent_prior not in (None, "flow"):
            raise ValueError(
                "Only the flow latent prior is supported. "
                "Use truncation rules to filter latent samples."
            )
        if latent_temperature is not None:
            if isinstance(latent_temperature, bool) or not isinstance(
                latent_temperature, (int, float)
            ):
                raise TypeError("latent_temperature must be a float")
            latent_temperature = float(latent_temperature)
            if latent_temperature <= 0.0:
                raise ValueError("latent_temperature must be positive")

        self.drawsize = drawsize
        self.latent_prior = "flow"
        self.latent_temperature = latent_temperature
        self.clip_population_weights = clip_population_weights

    def _get_population_log_weights(self, log_weights) -> np.ndarray:
        """Return log-weights used in the rejection step during population."""
        log_weights = np.asarray(log_weights, dtype=float)
        log_weights = log_weights - np.nanmax(log_weights)
        if not self.clip_population_weights:
            return log_weights

        weights = _clip_weights(np.exp(log_weights))
        return np.log(weights) - np.log(weights.max())

    def configure_truncation(
        self,
        truncation_method=None,
        truncation_methods=None,
        truncation_kwargs=None,
        truncate_log_q=False,
        enforce_likelihood_threshold=False,
        latent_radius_kwargs=None,
        default_latent_radius=False,
    ) -> None:
        use_default_latent_radius = default_latent_radius and (
            truncation_method is None and truncation_methods is None
        )
        existing_rule = self._get_latent_radius_rule()
        if (
            latent_radius_kwargs is None
            and truncation_method is None
            and truncation_methods is None
            and existing_rule is not None
        ):
            latent_radius_kwargs = existing_rule.to_kwargs()

        methods = build_truncation_methods(
            truncation_method=truncation_method,
            truncation_methods=truncation_methods,
            truncate_log_q=truncate_log_q,
            enforce_likelihood_threshold=enforce_likelihood_threshold,
            latent_radius_kwargs=latent_radius_kwargs,
            default_latent_radius=use_default_latent_radius,
        )
        self.truncation_method = truncation_method

        truncation_kwargs = normalise_truncation_kwargs(
            truncation_method=truncation_method,
            truncation_methods=truncation_methods,
            truncation_kwargs=truncation_kwargs,
        )
        if latent_radius_kwargs is not None and "latent_radius" in methods:
            truncation_kwargs["latent_radius"] = {
                **latent_radius_kwargs,
                **truncation_kwargs.get("latent_radius", {}),
            }

        methods, self.truncation_kwargs = apply_default_truncation_config(
            methods,
            truncation_kwargs=truncation_kwargs,
            default_latent_radius=use_default_latent_radius,
        )

        rules = []
        for method in methods:
            rule_cls = get_truncation_rule_class(method)
            raw_kwargs = self.truncation_kwargs.get(method, {})
            if not isinstance(raw_kwargs, dict):
                raise TypeError(
                    f"Truncation kwargs for {method} must be a dictionary"
                )

            kwargs = dict(raw_kwargs)
            rules.append(rule_cls(**kwargs))

        self._truncation_scheme = TruncationScheme(rules)
        self._sync_truncation_state()

    def set_rescaling(self) -> None:
        """Set the rescaling functions."""
        super().set_rescaling()
        self._truncation_scheme.configure(self)

    def backward_pass(
        self,
        z,
        rescale=True,
        discard_nans=True,
        return_z=False,
        return_unit_hypercube=False,
    ):
        """Apply the inverse flow and inverse rescaling."""
        try:
            x, log_j = self.flow.inverse(z)
            log_prob = self.latent_log_prob(z, self.latent_temperature) - log_j
        except AssertionError as e:
            logger.warning(
                "Assertion error raised when sampling from the flow. Error: %s",
                e,
            )
            if return_z:
                return np.array([]), np.array([]), np.array([])
            return np.array([]), np.array([])

        if discard_nans:
            valid = np.isfinite(log_prob)
            x, log_prob, z = get_subset_arrays(valid, x, log_prob, z)

        x_array = np.asarray(x, dtype=config.livepoints.default_float_dtype)
        if x_array.ndim == 1:
            x_array = x_array[np.newaxis, :]
        x = empty_structured_array(
            x_array.shape[0], dtype=self.x_prime_internal_dtype
        )
        for i, parameter in enumerate(self.prime_parameters):
            x[parameter] = x_array[:, i]

        if rescale:
            x, log_J = self.inverse_rescale(
                x, return_unit_hypercube=return_unit_hypercube
            )
            log_prob -= log_J
            if not return_unit_hypercube:
                x, z, log_prob = self.check_prior_bounds(x, z, log_prob)

        if return_z:
            return x, log_prob, z
        return x, log_prob

    def populate(
        self,
        worst_point,
        n_samples=10000,
        plot=True,
        r=None,
        max_samples=1_000_000,
    ) -> None:
        """Populate a pool of samples using staged truncation."""
        st = datetime.datetime.now()
        if not self.initialised:
            raise RuntimeError(
                "Proposal has not been initialised. Try calling `initialise()` "
                "first."
            )

        self._truncation_scheme.prepare(self, worst_point, radius=r)

        if self.indices:
            logger.debug(
                "Existing pool of samples is not empty. Discarding existing "
                "samples."
            )
        self.indices = []

        if self.accumulate_weights:
            samples = empty_structured_array(0, dtype=self.population_dtype)
        else:
            samples = empty_structured_array(
                n_samples, dtype=self.population_dtype
            )

        log_n = np.log(n_samples)
        log_n_expected = -np.inf
        n_proposed = 0
        log_weights = np.empty(0)
        n_accepted = 0
        accept = None
        hit_max_samples = False

        while n_accepted < n_samples:
            z = self.sample_latent_distribution(self.drawsize)
            n_proposed += z.shape[0]
            z = self._truncation_scheme.apply_latent(self, z)
            if not len(z):
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break
                continue

            x, log_q, z = self.backward_pass(
                z,
                rescale=True,
                return_z=True,
                return_unit_hypercube=self.map_to_unit_hypercube,
            )
            x, log_q, z = self._truncation_scheme.apply_after_backward(
                self, x, log_q, z
            )
            if not len(x):
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break
                continue

            if self._truncation_scheme.requires_log_likelihood:
                x["logL"] = self.model.batch_evaluate_log_likelihood(
                    x, unit_hypercube=self.map_to_unit_hypercube
                )
                x, log_q, z = self._truncation_scheme.apply_after_likelihood(
                    self, x, log_q, z
                )
                if not len(x):
                    if n_proposed > max_samples:
                        logger.warning("Reached max samples (%s)", max_samples)
                        break
                    continue

            log_w = self.compute_weights(x, log_q)

            if self.accumulate_weights:
                samples = np.concatenate([samples, x])
                log_weights = np.concatenate([log_weights, log_w])
                log_weights_rejection = self._get_population_log_weights(
                    log_weights
                )
                log_n_expected = logsumexp(log_weights_rejection)

                logger.debug(
                    "Drawn %s - n expected: %s / %s",
                    samples.size,
                    np.exp(log_n_expected),
                    n_samples,
                )

                if log_n_expected >= log_n:
                    log_u = np.log(self.rng.random(len(log_weights)))
                    accept = log_weights_rejection > log_u
                    n_accepted = np.sum(accept)
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    hit_max_samples = True
                    break
            else:
                log_w = self._get_population_log_weights(log_w)
                log_u = np.log(self.rng.random(len(log_w)))
                accept = log_w > log_u
                n_accept_batch = accept.sum()
                m = min(n_samples - n_accepted, n_accept_batch)
                samples[n_accepted : n_accepted + m] = x[accept][:m]
                n_accepted += n_accept_batch
                logger.debug("n accepted: %s / %s", n_accepted, n_samples)
                if n_proposed > max_samples:
                    logger.warning("Reached max samples (%s)", max_samples)
                    break

        if self.accumulate_weights:
            if accept is None or len(accept) != len(samples):
                log_weights_rejection = self._get_population_log_weights(
                    log_weights
                )
                log_u = np.log(self.rng.random(len(log_weights)))
                accept = log_weights_rejection > log_u
            logger.debug("Total number of samples: %s", samples.size)
            self.x = samples[accept][:n_samples]
            n_accepted = self.x.size
        else:
            self.x = samples[: min(n_samples, n_accepted)]
            n_accepted = self.x.size

        self.population_time += datetime.datetime.now() - st
        completed = n_accepted >= n_samples
        result = self.record_population_result(
            completed=completed,
            n_requested=n_samples,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            hit_max_samples=hit_max_samples,
            request_reset=hit_max_samples and n_proposed > 0,
        )

        if n_accepted == 0:
            self.x = None
            self.samples = None
            self.indices = []
            self.populated = False
            self._checked_population = True
            return result

        self.samples = self.convert_to_samples(self.x, plot=plot)
        if self._plot_pool and plot:
            self.plot_pool(self.samples)

        if not self._truncation_scheme.requires_log_likelihood:
            logger.debug("Evaluating log-likelihoods")
            self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
                self.samples
            )
        if self.check_acceptance:
            self.acceptance.append(
                self.compute_acceptance(worst_point["logL"])
            )
            logger.debug("Current acceptance %s", self.acceptance[-1])

        self.indices = self.rng.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        self._checked_population = False
        return result

    def reset(self) -> None:
        """Reset the proposal."""
        super().reset()
        self._truncation_scheme.reset()

    def __getstate__(self):
        return super().__getstate__()
