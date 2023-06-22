#!/usr/bin/env python

# Example of using the importance nested sampler in nessai with Neural
# spline flows defined on the unit-hypercube
#
# This example should take around 5 minutes to run

import numpy as np
import os

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = os.path.join("outdir", "nsf_unit_hypercube")
logger = setup_logger(output=output, log_level="INFO")

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


# Configure the flow to be a Neural Spline Flow with a uniform latent
# distribution
flow_config = dict(
    model_config=dict(
        n_blocks=4,
        n_neurons=32,
        ftype="nsf",
        distribution="uniform",
        kwargs=dict(
            linear_transform=None,
            batch_norm_between_layers=False,
            tail_bound=1.0,
            tails=None,
            num_bins=8,
        ),
    ),
)

# Set up the FlowSampler
# We set `reparam=None` because the flow is defined on the unit-hypercube
fs = FlowSampler(
    RosenbrockModel(4),
    nlive=10000,
    output=output,
    resume=False,
    seed=1234,
    importance_nested_sampler=True,
    draw_constant=True,
    reparameterisation=None,
    threshold_kwargs={"q": 0.66},
    reset_flow=4,
    flow_config=flow_config,
)

# And go!
fs.run()
