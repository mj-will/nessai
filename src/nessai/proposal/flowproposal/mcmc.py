from __future__ import annotations

import datetime
import os
from typing import Callable

import numpy as np

from ...livepoint import live_points_to_array
from .base import BaseFlowProposal


class MiniPCNFlowProposal(BaseFlowProposal):
    """Version of FlowProposal that uses MiniPCN instead of rejection sampling.

    Parameters
    ----------
    model: Model
        Model with likelihood and prior
    n_steps : int
        Number of MCMC steps to take
    step_fn: str
        Step function to use. See the minipcn documentation for details.
    minipcn_kwargs : dict
        Dictionary of keyword arguments passed to :code:`minipcn.Sampler` and
        `Sampler.sample`.
    enforce_likelihood_threshold: bool
        Check the likelihood constraint when sampling. If false, samples are drawn
        from the prior using the flow.
    plot_history: bool
        Toggle plotting the MCMC history.
    """

    def __init__(
        self,
        model,
        n_steps: int,
        step_fn: str = "tpcn",
        minipcn_kwargs: dict | None = None,
        enforce_likelihood_threshold: bool = True,
        plot_history: bool = False,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.n_steps = n_steps
        self.step_fn = step_fn
        self.minipcn_kwargs = minipcn_kwargs or {}
        self.enforce_likelihood_threshold = enforce_likelihood_threshold
        self.mcmc_history = {
            "acceptance": [],
            "n_steps": [],
        }
        self._plot_history = plot_history

    def _get_log_prob(
        self, logl_threshold: float
    ) -> Callable[[np.ndarray], float]:
        """Get the log-probability function for MiniPCN."""

        def _log_prob(z):
            """Log-probability for minipcn"""
            self.backward_pass(z, rescale=True, return_unit_hypercube=False)
            x, log_j_flow = self.backward_pass(
                z,
                rescale=True,
                return_unit_hypercube=self.map_to_unit_hypercube,
            )
            if self.map_to_unit_hypercube:
                log_p = self.unit_hypercube_log_prior(x)
            else:
                log_p = self.log_prior(x)
            finite_prior = np.isfinite(log_p)

            if self.enforce_likelihood_threshold:
                x["logL"][finite_prior] = (
                    self.model.batch_evaluate_log_likelihood(
                        x[finite_prior],
                        unit_hypercube=self.map_to_unit_hypercube,
                    )
                )
                above_threshold = x["logL"] > logl_threshold
                log_p[~above_threshold] = -np.inf

            val = log_p + log_j_flow
            return val

        return _log_prob

    def plot_history(self):
        """Plot the history of MCMC acceptance and number of steps.

        This is useful for diagnosing the performance of the MCMC proposal over
        the course of the run.
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(self.mcmc_history["acceptance"])
        axs[0].set_ylabel("Acceptance")
        axs[1].plot(self.mcmc_history["n_steps"])
        axs[1].set_ylabel("Number of steps")
        axs[-1].set_xlabel("Iteration")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output, "mcmc_history.png"))
        plt.close(fig)

    def populate(
        self,
        worst_point: np.ndarray,
        n_samples: int | None = 10000,
        plot: bool = True,
    ) -> None:
        """Populate the proposal pool using MiniPCN MCMC.

        Parameters
        ----------
        worst_point : np.ndarray
            The current worst point in the nested sampling run. Used to set the
            likelihood threshold.
        n_samples : int
            Number of samples to generate. If None, uses the poolsize.
        plot : bool
            Whether to plot diagnostic plots. Each plot can be toggled on/off
            use the corresponding keyword argument when initialising the proposal.
        """
        from minipcn import Sampler

        st = datetime.datetime.now()

        log_prob_fn = self._get_log_prob(worst_point["logL"])

        kwargs = self.minipcn_kwargs.copy()
        verbose = kwargs.pop("verbose", False)

        sampler = Sampler(
            log_prob_fn=log_prob_fn,
            step_fn=self.step_fn,
            rng=self.rng,
            dims=self.rescaled_dims,
            **kwargs,
        )

        x_prime_array = live_points_to_array(
            self.training_data_prime,
            self.prime_parameters,
            copy=True,
        )
        z_init, _ = self.flow.forward_and_log_prob(x_prime_array)

        n_walkers = n_samples if n_samples is not None else self.poolsize

        z_init = self.rng.choice(z_init, n_walkers, axis=0)
        chain, history = sampler.sample(
            z_init,
            n_steps=self.n_steps,
            verbose=verbose,
        )

        if self._plot_history and plot:
            self.plot_history()
        if self._plot_pool and plot:
            self.plot_pool(self.samples)

        z_pool = chain[-1]
        x_pool, _ = self.backward_pass(
            z_pool,
            rescale=True,
            return_unit_hypercube=False,
        )
        # This could be made more efficient
        x_pool["logL"] = self.model.batch_evaluate_log_likelihood(x_pool)
        self.samples = self.convert_to_samples(x_pool)

        self.mcmc_history["acceptance"].append(
            np.mean(history.acceptance_rate)
        )
        self.mcmc_history["n_steps"].append(self.n_steps)

        self.population_time += datetime.datetime.now() - st

        self.population_acceptance = np.mean(history.acceptance_rate)
        self.indices = self.rng.permutation(self.samples.size).tolist()
        self.populated_count += 1
        self.populated = True
        self._checked_population = False
