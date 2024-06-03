import datetime
from functools import lru_cache
import logging
import numpy as np

from ...proposal.flowproposal import FlowProposal
from ...livepoint import (
    live_points_to_array,
    numpy_array_to_live_points,
)
from ... import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix with size n, excluding the diagonal.

    This is direct copy the same function from emcee.
    """
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    return pairs


def differential_evolution_proposal(x, ensemble, sigma=1e-5):
    pairs = _get_nondiagonal_pairs(ensemble.shape[0])
    indices = np.random.choice(pairs.shape[0], size=x.shape[0], replace=True)
    diffs = np.diff(ensemble[pairs[indices]], axis=1).squeeze(axis=1)
    n = x.shape[0]
    g0 = 2.38 / np.sqrt(2 * x.shape[1])
    gamma = g0 * (1 + sigma * np.random.randn(n, 1))
    x_new = x + gamma * diffs
    return x_new


def gaussian_proposal(x, ensemble, sigma=1e-1):
    return x + sigma * np.random.randn(*x.shape)


class FlowProposalMCMC(FlowProposal):
    """Version of FlowProposal that uses MCMC instead of rejection sampling"""

    def __init__(self, *args, n_steps=50, **kwargs):
        self.n_steps = n_steps
        super().__init__(*args, **kwargs)

    def backward_pass(self, z: np.ndarray, rescale: bool = True):
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
            Samples in the data space
        log_j : array_like
            Determinant of the log-Jacobian
        """
        x, log_j = self.flow.inverse(z)

        x = numpy_array_to_live_points(
            x.astype(config.livepoints.default_float_dtype),
            self.rescaled_names,
        )
        if rescale:
            x, log_j_rescale = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_j += log_j_rescale
        return x, log_j

    def populate(
        self,
        worst_point,
        N=1000,
        plot=True,
    ):
        st = datetime.datetime.now()
        if not self.initialised:
            raise RuntimeError(
                "Proposal has not been initialised. "
                "Try calling `initialise()` first."
            )

        log_l_threshold = worst_point["logL"].copy()

        # Ensemble points
        x_prime_array = live_points_to_array(
            self.training_data_prime,
            self.rescaled_names,
            copy=True,
        )
        z_ensemble, _ = self.flow.forward_and_log_prob(x_prime_array)

        n_walkers = min(self.poolsize, self.training_data.size)

        # Initial points
        z_current = z_ensemble[:n_walkers]
        x_current, log_j_current = self.backward_pass(z_current)
        x_current["logP"] = self.model.batch_evaluate_log_prior(x_current)
        x_current["logL"] = self.model.batch_evaluate_log_likelihood(x_current)

        for _ in range(self.n_steps):
            z_new = differential_evolution_proposal(
                z_current,
                z_ensemble,
            )
            x_new, log_j_new = self.backward_pass(z_new, rescale=True)
            x_new["logP"] = self.model.batch_evaluate_log_prior(x_new)
            x_new["logL"] = self.model.batch_evaluate_log_likelihood(x_new)

            log_factor = (
                x_new["logP"] + log_j_new - x_current["logP"] - log_j_current
            )
            log_u = np.log(np.random.rand(n_walkers))
            accept = (
                (log_factor > log_u)
                & np.isfinite(x_new["logP"])
                & (x_new["logL"] > log_l_threshold)
            )

            x_current[accept] = x_new[accept]
            z_current[accept] = z_new[accept]
            log_j_current[accept] = log_j_new[accept]

        self.samples = self.convert_to_samples(x_current)

        self.population_time += datetime.datetime.now() - st
        if self._plot_pool and plot:
            self.plot_pool(self.samples)
        self.population_time += datetime.datetime.now() - st
        self.population_acceptance = np.nan
        n_above = np.sum(self.samples["logL"] > log_l_threshold)
        logger.info(f"n above threshold: {n_above} / {n_walkers}")
        self.indices = np.random.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        self._checked_population = False
