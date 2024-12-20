#!/usr/bin/env python

# Example of using the experimental MCMCFlowProposal

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import configure_logger

output = "./outdir/mcmc_example_cvm_8d_flow_act/"
logger = configure_logger(output=output)


class RosenbrockModel(Model):
    """Rosenbrock likelihood defined in n dimensions on [-5, 5]^n."""

    def __init__(self, dims):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        # The MCMC sampler can return samples outside the prior so we suppress
        # the warnings
        with np.errstate(divide="ignore"):
            log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        # Unstructured view returns a view of the inputs as a "normal" numpy
        # array without any fields. It will included the parameters listed in
        # names and in the same order. This is faster then converting from
        # live points to an array.
        x = self.unstructured_view(x)
        return -np.sum(
            100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
            + (1.0 - x[..., :-1]) ** 2.0,
            axis=-1,
        )


# The MCMCFlowProposal class shares most of the configuration options with
# FlowProposal but also has some specific options
# Here we also set
#  - n_accept : the number of accepted jumps required to stop
#  - reset_flow : this is recommended with the MCMC proposal
fs = FlowSampler(
    RosenbrockModel(8),
    output=output,
    resume=False,
    seed=1234,
    flow_proposal_class="mcmcflowproposal",
    plot_history=True,
    plot_chain=False,
    step_type="diff",
    enforce_likelihood_threshold=True,
    # constant_volume_fraction=0.98,
    n_steps=5000,
    reset_flow=16,
    flow_config=dict(
        n_neurons=32,
        n_blocks=8,
        n_layers=2,
        batch_norm_between_layers=True,
        linear_transform=None,
        net="mlp",
    ),
)

# And go!
fs.run()
