# -*- coding: utf-8 -*-
"""
Proposals specifically for use with the importance based nested sampler.
"""
import logging
from functools import partial
import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.special import logsumexp

from nessai.plot import plot_1d_comparison, plot_histogram, plot_live_points
from nessai.utils.testing import assert_structured_arrays_equal

from .base import Proposal
from .. import config
from ..flowmodel.importance import ImportanceFlowModel
from ..flowmodel.utils import update_config
from ..livepoint import (
    get_dtype,
    live_points_to_array,
    numpy_array_to_live_points,
    empty_structured_array,
)
from ..model import Model
from ..utils.rescaling import (
    logit,
    sigmoid,
)
from ..utils.structures import get_subset_arrays


logger = logging.getLogger(__name__)


class ImportanceFlowProposal(Proposal):
    """Flow-based proposal for importance-based nested sampling.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-defined model.
    output : str
        Output directory.
    flow_config : dict
        Configuration for the flow.
    reparameterisation : str
        Reparameterisation to use. If None, only the unit-hypercube
        reparameterisation is used.
    weighted_kl : bool
        If true, use the weights (prior / meta-proposal) when training the
        next flow.
    rest_flow : Union[bool, int]
        When to reset the flow. If False, the flow is not reset. If True the
        flow is reset every time a new flow is added. If an integer, every
        nth flow is reset.
    clip : bool
        If true the samples generated by flow will be clipped to [0, 1] before
        being mapped back from the unit-hypercube. This is only needed when
        the mapping cannot be defined outside of [0, 1]. In cases where it
        can, these points will be rejected when the prior bounds are checked.
    plot_training : bool
        If True, produce plots during training. If False, no plots are
        produced.
    """

    def __init__(
        self,
        model: Model,
        output: str,
        flow_config: dict = None,
        reparameterisation: str = "logit",
        weighted_kl: bool = True,
        reset_flow: Union[bool, int] = True,
        clip: bool = False,
        plot_training: bool = False,
    ) -> None:
        self._proposal_count = -1
        self._initialised = False

        self.model = model
        self.output = output
        self.flow_config = flow_config
        self.plot_training = plot_training
        self.reset_flow = int(reset_flow)
        self.reparameterisation = reparameterisation
        self.weighted_kl = weighted_kl
        self.clip = clip
        self._weights = {"-1": 1.0}

        self.dtype = get_dtype(self.model.names)

    @property
    def qid(self):
        return str(self._proposal_count)

    @property
    def weights(self) -> dict:
        """Dictionary containing the weights for each proposal"""
        return self._weights

    @property
    def log_q_dtype(self) -> np.dtype:
        return np.dtype([(qid, "f8") for qid in self.weights.keys()])

    @property
    def weights_array(self) -> np.ndarray:
        """Array of weights for each proposal"""
        return np.fromiter(self._weights.values(), dtype=float)

    @property
    def n_proposals(self) -> int:
        """Current number of proposals in the meta proposal"""
        return len(self.weights)

    @property
    def flow_config(self) -> dict:
        """Return the configuration for the flow"""
        return self._flow_config

    @flow_config.setter
    def flow_config(self, config: dict) -> None:
        """Set configuration (includes checking defaults)"""
        if config is None:
            config = dict(model_config=dict())
        elif "model_config" not in config:
            config["model_config"] = dict()
        config["model_config"]["n_inputs"] = self.model.dims
        self._flow_config = update_config(config)

    @property
    def _reset_flow(self) -> bool:
        """Boolean to indicate if the flow should be reset"""
        if not self.reset_flow or self._proposal_count % self.reset_flow:
            return False
        else:
            return True

    @staticmethod
    def _check_fields():
        """Check that the logQ and logW fields have been added."""
        if "logQ" not in config.livepoints.non_sampling_parameters:
            raise RuntimeError(
                "logQ field missing in non-sampling parameters."
            )
        if "logW" not in config.livepoints.non_sampling_parameters:
            raise RuntimeError(
                "logW field missing in non-sampling parameters."
            )
        if "logU" not in config.livepoints.non_sampling_parameters:
            raise RuntimeError(
                "logU field missing in non-sampling parameters."
            )

    def initialise(self):
        """Initialise the proposal"""
        self._check_fields()
        if self.initialised:
            logger.debug("Proposal already initialised")
            return

        self.verify_rescaling()

        self.flow = ImportanceFlowModel(
            config=self.flow_config, output=self.output
        )
        self.flow.initialise()
        super().initialise()

    def verify_rescaling(
        self, n: int = 1000, rtol: float = 1e-08, atol: float = 1e-08
    ) -> None:
        """Verify the rescaling is invertible.

        Uses :code:`numpy.allclose`, see numpy documentation for more details.

        Parameters
        ----------
        n : int
            Number of samples to test.
        atol : float
            The absolute tolerance.
        rtol : float
            The relative tolerance.
        """
        logger.debug("Verifying rescaling")
        x_in = self.model.sample_unit_hypercube(n)

        x_prime, log_j = self.rescale(x_in)
        x_re, log_j_inv = self.inverse_rescale(x_prime)

        try:
            assert_structured_arrays_equal(x_re, x_in, atol=atol, rtol=rtol)
        except AssertionError as e:
            raise RuntimeError(f"Rescaling is not invertible. Error: {e}")

        if not np.allclose(log_j, -log_j_inv, atol=atol, rtol=rtol):
            raise RuntimeError(
                "Forward and inverse Jacobian determinants are not equal"
            )
        logger.debug("Rescaling functions are invertible")

    def to_prime(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples from the unit hypercube to samples in x'-space

        Parameters
        ----------
        x_prime :
            Unstructured array of samples in the unit hypercube

        Returns
        -------
        x :
            Unstructured array of samples in x'-space
        log_j :
            Corresponding log-Jacobian determinant.
        """
        x = np.atleast_2d(x)
        if self.reparameterisation == "logit":
            x_prime, log_j = logit(x, eps=config.general.eps)
            log_j = log_j.sum(axis=1)
        elif self.reparameterisation is None:
            x_prime = x.copy()
            log_j = np.zeros(x.shape[0])
        else:
            raise ValueError(
                f"Unknown reparameterisation: '{self.reparameterisation}'"
            )
        return x_prime, log_j

    def from_prime(self, x_prime: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples the x'-space to samples in the unit hypercube.

        Parameters
        ----------
        x_prime :
            Unstructured array of samples.

        Returns
        -------
        x :
            Unstructured array of samples in the unit hypercube.
        log_j :
            Corresponding log-Jacobian determinant.
        """
        x_prime = np.atleast_2d(x_prime)
        if self.reparameterisation == "logit":
            x, log_j = sigmoid(x_prime)
            log_j = log_j.sum(axis=1)
        elif self.reparameterisation is None:
            x = x_prime.copy()
            log_j = np.zeros(x.shape[0])
        else:
            raise ValueError(
                f"Unknown reparameterisation: '{self.reparameterisation}'"
            )
        return x, log_j

    def rescale(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to the space in which the flow is trained.

        Returns an unstructured array.
        """
        x = live_points_to_array(x, self.model.names)
        x_prime, log_j = self.to_prime(x)
        return x_prime, log_j

    def inverse_rescale(self, x_prime: np.ndarray) -> np.ndarray:
        """Convert from the space in which the flow is trained.

        Returns a structured array.
        """
        x, log_j = self.from_prime(x_prime)
        if self.clip:
            x = np.clip(x, 0.0, 1.0)
        x = numpy_array_to_live_points(x, self.model.names)
        return x, log_j

    def update_proposal_weights(self, weights: dict) -> None:
        """Method to update the proposal weights dictionary.

        Raises
        ------
        RuntimeError
            If the weights do not sum to 1 are the update.
        """
        self._weights.update(weights)
        w_sum = np.sum(np.fromiter(self._weights.values(), float))
        if not np.isclose(w_sum, 1.0):
            raise RuntimeError(f"Weights must sum to 1! Actual value: {w_sum}")

    def train(
        self,
        samples: np.ndarray,
        plot: bool = False,
        output: Union[str, None] = None,
        weights: np.ndarray = None,
        **kwargs,
    ) -> str:
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
        self._proposal_count += 1
        self._weights[self.qid] = np.nan
        output = self.output if output is None else output
        level_output = os.path.join(output, f"level_{self.qid}", "")

        if not os.path.exists(level_output):
            os.makedirs(level_output, exist_ok=True)

        training_data = samples.copy()
        x_prime, _ = self.rescale(training_data)

        if plot:
            plot_live_points(
                training_data,
                filename=os.path.join(level_output, "training_data.png"),
            )
            plot_1d_comparison(
                x_prime,
                convert_to_live_points=True,
                filename=os.path.join(level_output, "prime_training_data.png"),
            )

        logger.debug(
            f"Training data min and max: {x_prime.min()}, {x_prime.max()}"
        )

        if self.weighted_kl or weights is not None:
            logger.debug("Using weights in training")
            if weights is not None:
                weights = weights / np.sum(weights)
            else:
                log_weights = training_data["logW"].copy()
                log_weights -= logsumexp(log_weights)
                weights = np.exp(log_weights)
            if np.isnan(weights).any():
                raise ValueError("Weights contain NaN(s)")
            if not np.isfinite(weights).all():
                raise ValueError("Weights contain Inf(s)")

            if plot:
                plot_histogram(
                    weights, filename=level_output + "training_weights.png"
                )
        else:
            weights = None

        self.flow.add_new_flow(reset=self._reset_flow)

        logger.debug(f"Training with {x_prime.shape[0]} samples")
        self.flow.train(
            x_prime,
            weights=weights,
            output=level_output,
            plot=plot or self.plot_training,
            **kwargs,
        )

        if plot:
            test_samples_prime, log_prob = self.flow.sample_and_log_prob(2000)
            test_samples, log_j_inv = self.inverse_rescale(test_samples_prime)
            log_prob -= log_j_inv
            test_samples["logQ"] = log_prob
            plot_live_points(
                test_samples,
                filename=os.path.join(level_output, "generated_samples.png"),
            )
        return self.qid

    def compute_log_Q(
        self,
        x_prime: np.ndarray,
        log_j: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the log meta proposal (log Q) for an array of points.

        Parameters
        ----------
        x_prime : numpy.ndarray
            Array of samples in the unit hypercube.
        log_j : Optional[numpy.ndarray]
            Log-Jacobian determinant of the prime samples. Must be supplied if
            proposal includes flows.

        Returns
        -------
        log_Q : numpy.ndarray
            Value of the meta-proposal in the prime space.
        log_q : numpy.ndarray
            Array of values for flow in the flow in the proposal. Array will
            have shape (# samples, # flows)
        """
        if np.isnan(x_prime).any():
            logger.warning("NaNs in samples when computing log_Q")
        if not np.isfinite(x_prime).all():
            logger.warning(
                "Infinite values in the samples when computing log_Q"
            )

        if any(np.isnan(w) for w in self.weights.values()):
            raise RuntimeError("Some weights are not set!")

        log_q = np.empty(len(x_prime), dtype=self.log_q_dtype)
        if self.n_proposals > 1 and log_j is None:
            raise RuntimeError(
                "Must specify log_j! Meta-proposal includes flows"
            )
        if any([flow.training for flow in self.flow.models.values()]):
            raise RuntimeError("One or more flows are in training mode!")
        assert log_j is not None

        for name in log_q.dtype.names:
            log_prob_fn = self.get_proposal_log_prob(name, log_j=log_j)
            log_q[name] = log_prob_fn(x_prime)

        log_Q = self.compute_meta_proposal_from_log_q(log_q)

        if np.isnan(log_Q).any():
            raise ValueError("There is a NaN in log Q!")

        return log_Q, log_q

    def draw(
        self,
        n: int,
        flow_number: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw n new points.

        Parameters
        ----------
        n : int
            Number of points to draw.
        flow_number : Optional[int]
            Specifies which flow to use. If not specified the last flow will
            be used.

        Returns
        -------
        np.ndarray :
            Array of new points.
        np.ndarray :
            Log-proposal probabilities (log_q)
        """
        if flow_number is None:
            flow_number = self.qid

        # Draw a few more samples in case some are not accepted.
        n_draw = int(1.01 * n)
        logger.debug(f"Drawing {n} points")
        samples = np.zeros(0, dtype=self.dtype)
        log_q_samples = np.empty(0, dtype=self.log_q_dtype)

        n_accepted = 0
        while n_accepted < n and n_draw > 0:
            logger.debug(f"Drawing batch of {n_draw} samples")
            # x_prime, log_q = self.flow.sample_and_log_prob(N=n_draw)
            x_prime = self.flow.sample_ith(i=flow_number, N=n_draw)
            x, log_j_inv = self.inverse_rescale(x_prime)
            # Rescaling can sometimes produce infs that don't appear in samples
            x_check, log_j = self.rescale(x)
            # Probably don't need all these checks.
            acc = (
                self.model.in_unit_hypercube(x)
                & np.isfinite(x_check).all(axis=1)
                & np.isfinite(x_prime).all(axis=1)
                & np.isfinite(log_j)
                & np.isfinite(log_j_inv)
            )
            logger.debug(f"Rejected {n_draw - acc.sum()} points")
            if not np.any(acc):
                continue
            x, x_prime, log_j = get_subset_arrays(acc, x, x_prime, log_j)

            x["logQ"], log_q = self.compute_log_Q(x_prime, log_j=log_j)
            x["logP"] = self.model.batch_evaluate_log_prior(
                x, unit_hypercube=True
            )
            x["logU"] = self.model.batch_evaluate_log_prior_unit_hypercube(x)
            x["logW"] = x["logU"] - x["logQ"]
            accept = np.isfinite(x["logP"]) & ~np.isposinf(x["logW"])
            if not np.any(accept):
                continue

            x, log_q = get_subset_arrays(accept, x, log_q)
            assert np.array_equal(
                x["logQ"], self.compute_meta_proposal_from_log_q(log_q)
            )
            samples = np.concatenate([samples, x])
            log_q_samples = np.concatenate([log_q_samples, log_q])
            n_accepted += x.size
            logger.debug(f"Accepted: {n_accepted}")

        samples = samples[:n]
        log_q_samples = log_q_samples[:n]
        samples["qID"] = self.qid
        logger.debug(
            f"Mean log_q for each each flow: {rfn.apply_along_fields(np.mean, log_q_samples)}"
        )

        logger.debug(f"Returning {samples.size} samples")
        return samples, log_q_samples

    def update_log_q(
        self,
        samples: np.ndarray,
        log_q: np.ndarray,
    ) -> np.ndarray:
        """Update the array of proposal probabilities for a set of samples"""
        if self.qid in log_q.dtype.names:
            raise ValueError("log_q array already contains current proposal")
        x, log_j = self.rescale(samples)
        log_prob_fn = self.get_proposal_log_prob(self.qid, log_j=log_j)
        log_q_current = log_prob_fn(x)
        log_q = rfn.append_fields(
            log_q,
            self.qid,
            log_q_current,
            usemask=False,
        )
        return log_q

    def compute_meta_proposal_from_log_q(self, log_q):
        """Compute the meta-proposal from an array of proposal \
        log-probabilities
        """
        return rfn.apply_along_fields(
            partial(logsumexp, b=self.weights_array),
            log_q,
        )

    def compute_meta_proposal_samples(self, samples: np.ndarray) -> np.ndarray:
        """Compute the meta proposal Q for a set of samples.

        Includes any rescaling that has been configured.

        Returns
        -------
        log_meta_proposal : numpy.ndarray
            Array of meta-proposal log probabilities (log Q)
        log_q : numpy.ndarray
            Array of log q for each flow.
        """
        if self.qid not in self.weights or np.isnan(self.weights[self.qid]):
            raise RuntimeError(
                "Weight(s) missing or not set. "
                f"Current weights: {self.weights}."
            )
        x, log_j = self.rescale(samples)
        return self.compute_log_Q(x, log_j=log_j)

    def get_proposal_log_prob(
        self, it: str, log_j: np.ndarray = None
    ) -> Callable:
        """Get a pointer to the function for ith proposal."""
        if it == "-1":
            return lambda x: np.zeros(len(x))
        elif it in self.flow.models:
            if log_j is not None:
                return lambda x: self.flow.log_prob_ith(x, it) + log_j
            else:
                return lambda x: self.flow.log_prob_ith(x, it)
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
            p_it = self.flow.n_models - 1

        if q_it is None:
            q_it = self.flow.n_models - 2

        if p_it == q_it:
            raise ValueError("p and q must be different")
        elif p_it < -1 or q_it < -1:
            raise ValueError(f"Invalid p_it or q_it: {p_it}, {q_it}")

        log_p_f = self.get_proposal_log_prob(p_it)
        log_q_f = self.get_proposal_log_prob(q_it)

        log_p = log_p_f(x_prime)
        log_q = log_q_f(x_prime)

        if p_it > -1:
            log_p += log_j
        if q_it > -1:
            log_q += log_j

        kl = np.mean(log_p - log_q)
        logger.info(f"KL between {p_it} and {q_it} is: {kl:.3}")
        return kl

    def draw_from_prior(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Draw from the prior"""
        samples = self.model.sample_unit_hypercube(n)
        samples["logU"] = self.model.batch_evaluate_log_prior_unit_hypercube(
            samples
        )
        prime_samples, log_j = self.rescale(samples)
        log_Q, log_q = self.compute_log_Q(prime_samples, log_j=log_j)
        samples["logQ"] = log_Q
        samples["logW"] = samples["logU"] - log_Q
        return samples, log_q

    def draw_from_flows(
        self,
        n: int,
        weights=None,
        counts=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw n points from all flows (g).

        Parameters
        ----------
        n : int
            Number of points
        weights
            Array of (normalised) weights. Ignored if counts is specified.
        counts
            Number of samples to draw from each proposal.
        """
        logger.debug(
            f"Drawing {n} samples from the combination of all the proposals"
        )
        if counts is None:
            if weights is None:
                weights = self.weights
            if not np.sum(weights) == 1:
                weights = weights / np.sum(weights)
            if not len(weights) == self.n_proposals:
                ValueError(
                    "Size of weights does not match the number of levels"
                )
            logger.debug(f"Proposal weights: {weights}")
            counts = np.random.multinomial(n, weights)
        else:
            counts = np.array(counts, dtype=int)
            weights = counts / counts.sum()
        logger.debug(f"Expected counts: {counts}")
        if np.any(counts) < 0:
            raise ValueError("Cannot have negative counts")
        if np.sum(counts) == 0:
            raise ValueError("Total counts is zero")
        proposal_id = np.arange(weights.size) - 1
        prime_samples = np.empty([n, self.model.dims])
        sample_its = np.empty(n, dtype=config.livepoints.it_dtype)
        count = 0
        # Draw from prior
        for id, m in zip(proposal_id, counts):
            if m == 0:
                continue
            logger.debug(f"Drawing {m} samples from the {id}th proposal.")
            if id == -1:
                prime_samples[count : (count + m)] = self.to_prime(
                    np.random.rand(m, self.model.dims)
                )[0]
            else:
                prime_samples[count : (count + m)] = self.flow.sample_ith(
                    id, N=m
                )
            sample_its[count : (count + m)] = id
            count += m

        samples, log_j_inv = self.inverse_rescale(prime_samples)
        samples["it"] = sample_its
        x_check, log_j = self.rescale(samples)
        # Probably don't need all these checks.
        finite = (
            self.model.in_unit_hypercube(samples)
            & np.isfinite(x_check).all(axis=1)
            & np.isfinite(prime_samples).all(axis=1)
            & np.isfinite(log_j)
            & np.isfinite(log_j_inv)
        )

        samples, prime_samples, log_j = get_subset_arrays(
            finite, samples, prime_samples, log_j
        )

        log_q = np.zeros((samples.size, self.n_proposals))
        logger.debug("Computing log_q")
        if self.n_proposals > 1:
            log_q[:, 1:] = (
                self.flow.log_prob_all(prime_samples) + log_j[:, np.newaxis]
            )

        # -inf is okay since this is just zero, so only remove +inf or NaN
        finite = ~np.isnan(log_q).all(axis=1) & ~np.isposinf(log_q).all(axis=1)
        samples, log_q = get_subset_arrays(finite, samples, log_q)

        logger.debug(
            f"Mean g for each each flow: {np.exp(log_q).mean(axis=0)}"
        )
        logger.debug(f"Mean log_q for each each flow: {log_q.mean(axis=0)}")

        samples["logP"] = self.model.batch_evaluate_log_prior(
            samples, unit_hypercube=True
        )
        samples, log_q = get_subset_arrays(
            np.isfinite(samples["logP"]), samples, log_q
        )
        counts = np.bincount(
            samples["it"] + 1,
            minlength=self.n_proposals,
        ).astype(int)
        logger.debug(f"Actual counts: {counts}")

        return samples, log_q, counts

    def remove_proposal(self, i: str) -> None:
        """Remove a proposal from the meta-proposal and update the weights"""
        self.flow.remove_flow(i)
        self._weights.pop(i)
        #self.weights = self.renormalise_weights(self.weights)

    @staticmethod
    def renormalise_weights(weights: dict) -> dict:
        """Renormalises the weights and updates the labels"""
        norm_constant = np.sum(np.fromiter(weights.values(), float))
        new_weights = {
            k - 1: v / norm_constant for k, v in enumerate(weights.values())
        }
        return new_weights

    def resume(self, model, flow_config, weights_path=None):
        """Resume the proposal"""
        super().resume(model)
        self.flow_config = flow_config
        self.initialise()
        self.flow.resume(
            self.flow_config["model_config"], weights_path=weights_path
        )

    def __getstate__(self):
        d = self.__dict__
        exclude = {"model", "_flow_config", "flow"}
        state = {k: d[k] for k in d.keys() - exclude}
        return state, self.flow

    def __setstate__(self, state):
        self.__dict__.update(state[0])
        self.flow = state[1]
