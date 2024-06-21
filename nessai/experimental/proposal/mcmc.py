import datetime
from functools import lru_cache
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

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


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`

    Copied from emcee.
    """
    i = 1
    while i < n:
        i = i << 1
    return i


def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Copied from emcee
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    if acf[0] == 0.0:
        acf[:] = np.nan_to_num(np.inf)
    else:
        acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def integrated_time(x, c=5):
    """Estimate the integrated autocorrelation time of a time series.

    Based on the implementation from emcee.
    """
    n_t, n_w, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += function_1d(x[:, k, d])
        f /= n_w
        taus = 2.0 * np.cumsum(f) - 1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    return tau_est


def differential_evolution_proposal(x, ensemble, sigma=1e-4, mix_fraction=0.8):
    pairs = _get_nondiagonal_pairs(ensemble.shape[0])
    indices = np.random.choice(pairs.shape[0], size=x.shape[0], replace=True)
    diffs = np.diff(ensemble[pairs[indices]], axis=1).squeeze(axis=1)
    # n = x.shape[0]
    mix = np.random.rand(x.shape[0]) < mix_fraction
    g0 = 2.38 / np.sqrt(2 * x.shape[1])
    scale = np.ones((x.shape[0], 1))
    scale[mix, :] = g0
    error = sigma * np.random.randn(*scale.shape)
    x_new = x + scale * diffs + error
    return x_new


def gaussian_proposal(x, ensemble, sigma=0.2):
    return x + sigma * np.random.randn(*x.shape)


class FlowProposalMCMC(FlowProposal):
    """Version of FlowProposal that uses MCMC instead of rejection sampling"""

    def __init__(
        self,
        *args,
        n_steps=10,
        proposal: str = "diff",
        use_approximate_likelihood=False,
        approximator_threshold: float = 0.5,
        **kwargs,
    ):
        self.n_steps = n_steps
        self.proposal = proposal
        self.use_approximate_likelihood = use_approximate_likelihood
        self.approximator_threshold = approximator_threshold

        if self.use_approximate_likelihood:
            from ..approximator import Approximator

            self.approximator = Approximator()

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

    def plot_chain(self, chains):

        nsteps, nchains, ndims = chains.shape

        fig, axs = plt.subplots(ndims, 1, figsize=(4, 10))
        for i in range(nchains):
            for j in range(ndims):
                axs[j].plot(chains[:, i, j])
        fig.savefig(
            os.path.join(self.output, f"chain_{self.populated_count}.png")
        )
        plt.close(fig)

    def train_approximator(self, samples, threshold):
        x_prime, _ = self.rescale(samples)
        x_prime_array = live_points_to_array(
            x_prime,
            self.rescaled_names,
            copy=True,
        )
        z_train, _ = self.flow.forward_and_log_prob(x_prime_array)
        self.approximator.train(z_train, samples["logL"], threshold)

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
        np.random.shuffle(x_prime_array)
        z_ensemble, _ = self.flow.forward_and_log_prob(x_prime_array)

        n_walkers = min(self.poolsize, self.training_data.size)

        # Initial points
        z_current = z_ensemble[:n_walkers]
        x_current, log_j_current = self.backward_pass(z_current)
        x_current["logP"] = self.model.batch_evaluate_log_prior(x_current)
        x_current["logL"] = self.model.batch_evaluate_log_likelihood(x_current)

        z_chain = np.empty((self.n_steps, n_walkers, z_current.shape[-1]))
        z_chain[0] = z_current

        z_new_history = []

        # if self.proposal == "diff":
        #     proposal_fn = differential_evolution_proposal
        # elif self.proposal == "gaussian":
        #     proposal_fn = gaussian_proposal

        for i in range(self.n_steps):

            a = np.random.rand()
            if a < 0.5:
                proposal_fn = gaussian_proposal
            else:
                proposal_fn = differential_evolution_proposal

            z_new = proposal_fn(
                z_current,
                z_ensemble,
            )
            z_new_history.append(z_new)

            x_new, log_j_new = self.backward_pass(z_new, rescale=True)
            x_new["logP"] = self.model.batch_evaluate_log_prior(x_new)
            finite_prior = np.isfinite(x_new["logP"])

            if self.use_approximate_likelihood:
                prob_above = self.approximator.predict_prob_class(z_new, 1)
                logl_accept = prob_above > self.approximator_threshold
            else:
                # Only evaluate function where log-prior is finite
                # Default is NaN, so will not pass threshold.
                x_new["logL"][finite_prior] = (
                    self.model.batch_evaluate_log_likelihood(
                        x_new[finite_prior]
                    )
                )
                logl_accept = x_new["logL"] > log_l_threshold
            log_factor = (
                x_new["logP"] + log_j_new - x_current["logP"] - log_j_current
            )
            log_u = np.log(np.random.rand(n_walkers))
            # print(log_factor[214], log_u[214], finite_prior[214], logl_accept[214])

            # print(sum(log_factor > log_u), finite_prior.sum(), logl_accept.sum())
            accept = (log_factor > log_u) & finite_prior & logl_accept
            # print(accept.sum())

            x_current[accept] = x_new[accept]
            z_current[accept] = z_new[accept]
            log_j_current[accept] = log_j_new[accept]

            z_chain[i] = z_current

        z_new_history = np.array(z_new_history)

        # for i in range(n_walkers):
        #     if np.all(z_chain[0, i] == z_chain[:, i]):
        #         print(i)
        #         # print(z_chain[:, i])
        #         print(log_j_current[i])
        #         # print(z_new_history[:, i])
        # exit()

        # act = integrated_time(z_chain)
        # logger.info(f"ACT: {act}")

        self.samples = self.convert_to_samples(x_current)

        if self.use_approximate_likelihood:
            self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
                self.samples
            )

        self.population_time += datetime.datetime.now() - st
        self.plot_chain(z_chain)
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
