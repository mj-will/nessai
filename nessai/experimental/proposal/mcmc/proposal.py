import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from ....proposal.flowproposal import FlowProposal
from ....livepoint import (
    live_points_to_array,
    numpy_array_to_live_points,
)
from .... import config

from .steps import (
    KNOWN_STEPS,
)

logger = logging.getLogger(__name__)


class MCMCFlowProposal(FlowProposal):
    """Version of FlowProposal that uses MCMC instead of rejection sampling"""

    def __init__(
        self,
        *args,
        n_steps=10,
        step_type: str = "diff",
        plot_chain: bool = False,
        **kwargs,
    ):
        self.n_steps = n_steps
        self.step_type = step_type
        self._plot_chain = plot_chain

        StepClass = KNOWN_STEPS.get(self.step_type)
        self.step = StepClass()

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

        self.step.update_ensemble(z_ensemble)

        n_walkers = min(self.poolsize, self.training_data.size)

        # Initial points
        z_current = z_ensemble[:n_walkers]
        x_current, log_j_current = self.backward_pass(z_current)
        x_current["logP"] = self.model.batch_evaluate_log_prior(x_current)
        x_current["logL"] = self.model.batch_evaluate_log_likelihood(x_current)

        z_chain = np.empty((self.n_steps, n_walkers, z_current.shape[-1]))
        z_chain[0] = z_current

        z_new_history = []

        for i in range(self.n_steps):

            z_new, log_j_step = self.step(z_current)
            z_new_history.append(z_new)

            x_new, log_j_flow = self.backward_pass(z_new, rescale=True)
            x_new["logP"] = self.model.batch_evaluate_log_prior(x_new)
            finite_prior = np.isfinite(x_new["logP"])

            log_j_new = log_j_step + log_j_flow

            # Only evaluate function where log-prior is finite
            # Default is NaN, so will not pass threshold.
            x_new["logL"][finite_prior] = (
                self.model.batch_evaluate_log_likelihood(x_new[finite_prior])
            )
            logl_accept = x_new["logL"] > log_l_threshold
            log_factor = (
                x_new["logP"] + log_j_new - x_current["logP"] - log_j_current
            )
            log_u = np.log(np.random.rand(n_walkers))

            accept = (log_factor > log_u) & finite_prior & logl_accept

            x_current[accept] = x_new[accept]
            z_current[accept] = z_new[accept]
            log_j_current[accept] = log_j_new[accept]

            z_chain[i] = z_current

        z_new_history = np.array(z_new_history)

        self.samples = self.convert_to_samples(x_current)

        self.population_time += datetime.datetime.now() - st
        if self._plot_chain:
            self.plot_chain(z_chain)
        if self._plot_pool and plot:
            self.plot_pool(self.samples)
        self.population_acceptance = np.nan
        n_above = np.sum(self.samples["logL"] > log_l_threshold)
        logger.info(f"n above threshold: {n_above} / {n_walkers}")
        self.indices = np.random.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        self._checked_population = False
