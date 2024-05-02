#!/usr/bin/env python

# Example of using the importance nested sampler in nessai

import numpy as np
import os

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import nessai_models

output = os.path.join("outdir", "rebalance")
logger = setup_logger(output=output, log_level="DEBUG")

# Define the model. For the importance nested sampler we must define mappings
# to and from the unit hyper-cube.


class RosenbrockModel(Model):
    """A Rosenbrock likelihood."""

    def __init__(self, dims):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        """Log-prior"""
        log_p = np.log(self.in_bounds(x), dtype=float)
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood"""
        x = self.unstructured_view(x)
        return -(
            np.sum(
                100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
                + (1.0 - x[..., :-1]) ** 2.0,
                axis=-1,
            )
        )

    def to_unit_hypercube(self, x):
        """Map to the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (x[n] - self.bounds[n][0]) / (
                self.bounds[n][1] - self.bounds[n][0]
            )
        return x_out

    def from_unit_hypercube(self, x):
        """Map from the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out


flow_config = dict(
    model_config=dict(
        n_blocks=8,
        n_neurons=32,
    )
)

fs = FlowSampler(
    nessai_models.SlabSpike(dims=5,
                            spike_scale = 1e-3,
                            ),
    nlive=5000,
    output=output,
    resume=False,
    seed=1234,
    importance_nested_sampler=True,  # Use the importance nested sampler
    draw_constant=True,  # Draw a constant number of samples (2000)
    rebalance_interval=2,
    flow_config=flow_config,
    reset_flow=1,
    plot=False
)

# And go!
fs.run(redraw_samples=True, optimise_weights=True, optimisation_method="kl")
