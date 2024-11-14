import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from ....livepoint import (
    live_points_to_array,
)
from ....proposal.flowproposal.base import BaseFlowProposal
from .steps import (
    KNOWN_STEPS,
)

logger = logging.getLogger(__name__)


class MCMCFlowProposal(BaseFlowProposal):
    """Version of FlowProposal that uses MCMC instead of rejection sampling"""

    def __init__(
        self,
        model,
        n_steps=10,
        n_accept=None,
        step_type: str = "gaussian",
        plot_chain: bool = False,
        **kwargs,
    ):
        self.n_steps = n_steps
        self.n_accept = n_accept
        self.step_type = step_type
        self._plot_chain = plot_chain
        super().__init__(model, **kwargs)

    def initialise(self, resumed: bool = False):
        super().initialise(resumed=resumed)
        StepClass = KNOWN_STEPS.get(self.step_type)
        self.step = StepClass(dims=self.rescaled_dims, rng=self.rng)

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

    def x_prime_log_prior(self, x):
        raise RuntimeError(
            "MCMCFlowProposal does not support using x-prime priors"
        )

    def populate(
        self,
        worst_point,
        n_samples=1000,
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
            self.prime_parameters,
            copy=True,
        )
        self.rng.shuffle(x_prime_array)
        z_ensemble, _ = self.flow.forward_and_log_prob(x_prime_array)

        self.step.update_ensemble(z_ensemble)

        poolsize = n_samples if n_samples is not None else self.poolsize
        n_walkers = min(poolsize, self.training_data.size)

        # Initial points
        z_current = z_ensemble[:n_walkers]
        x_current, log_j_current = self.backward_pass(
            z_current, return_unit_hypercube=self.map_to_unit_hypercube
        )
        if self.map_to_unit_hypercube:
            log_p_current = self.unit_hypercube_log_prior(x_current)
        else:
            log_p_current = self.log_prior(x_current)
        x_current["logL"] = self.model.batch_evaluate_log_likelihood(
            x_current, unit_hypercube=self.map_to_unit_hypercube
        )

        z_chain = np.empty((self.n_steps, n_walkers, z_current.shape[-1]))
        z_chain[0] = z_current

        z_new_history = []
        n_accept = np.zeros(n_walkers)
        n_reject = np.zeros(n_walkers)

        for i in range(self.n_steps):
            z_new, log_j_step = self.step(z_current)
            z_new_history.append(z_new)

            x_new, log_j_flow = self.backward_pass(
                z_new,
                rescale=True,
                return_unit_hypercube=self.map_to_unit_hypercube,
            )
            if self.map_to_unit_hypercube:
                log_p = self.unit_hypercube_log_prior(x_new)
            else:
                log_p = self.log_prior(x_new)
            finite_prior = np.isfinite(log_p)

            log_j_new = log_j_step + log_j_flow

            # Only evaluate function where log-prior is finite
            # Default is NaN, so will not pass threshold.
            x_new["logL"][finite_prior] = (
                self.model.batch_evaluate_log_likelihood(
                    x_new[finite_prior],
                    unit_hypercube=self.map_to_unit_hypercube,
                )
            )
            logl_accept = x_new["logL"] > log_l_threshold
            log_factor = log_p + log_j_new - log_p_current - log_j_current
            log_u = np.log(self.rng.random(n_walkers))

            accept = (log_factor > log_u) & finite_prior & logl_accept

            x_current[accept] = x_new[accept]
            z_current[accept] = z_new[accept]
            log_j_current[accept] = log_j_new[accept]
            n_accept += accept
            n_reject += 1 - accept
            z_chain[i] = z_current
            if self.n_accept is not None and n_accept.mean() > self.n_accept:
                n_steps = i
                break
        else:
            n_steps = self.n_steps
            if self.n_accept is not None:
                logger.warning(
                    (
                        f"Reached max steps ({self.n_steps}) with "
                        f"n_accept={self.n_accept}!"
                    )
                )

        keep = n_accept > 0
        logger.debug(f"Replacing {n_walkers - keep.sum()} walkers")
        x_current = x_current[keep]

        z_new_history = np.array(z_new_history)
        z_chain = z_chain[:n_steps]
        self.step.update_stats(
            n_accept=n_accept.mean(),
            n_reject=n_reject.mean(),
        )

        self.population_time += datetime.datetime.now() - st
        if len(x_current) == 0:
            logger.warning("No samples accepted!")
            return

        self.samples = self.convert_to_samples(x_current)
        if self._plot_chain and plot:
            self.plot_chain(z_chain)
        if self._plot_pool and plot:
            self.plot_pool(self.samples)
        self.population_acceptance = n_accept.mean() / (
            n_accept.mean() + n_reject.mean()
        )
        logger.debug(f"MCMC acceptance: {self.population_acceptance}")
        self.indices = self.rng.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        self._checked_population = False
