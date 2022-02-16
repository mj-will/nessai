#!/usr/bin/env python

# Example of using nessai for a Rosenbrock likelihood.
# Also shows how to configure the normalising flow.

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.livepoint import live_points_to_array

output = './outdir/rosenbrock/'
logger = setup_logger(output=output)


class RosenbrockModel(Model):
    """Rosenbrock function defined in n dimensions on [-5, 5]^n.

    Based on the example in cpnest:\
        https://github.com/johnveitch/cpnest/blob/master/examples/rosenbrock.py
    """
    def __init__(self, dims):
        self.names = [f'x_{d}' for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        """Uniform prior"""
        log_p = np.log(self.in_bounds(x))
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood"""
        x = live_points_to_array(x, self.names)
        return -np.sum(
            100. * (x[..., 1:] - x[..., :-1] ** 2.) ** 2.
            + (1. - x[..., :-1]) ** 2.,
            axis=-1
        )


model = RosenbrockModel(5)

# The Rosenbrock likelihood is more complex, so we configure the normalsing
# flow to improve nessai's performance.
flow_config = dict(
    model_config=dict(
        n_blocks=4,
        n_neurons=10,
        n_layers=3
    )
)

# Configure the sampler.
fs = FlowSampler(
    model,
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=1451,
)

# And go!
fs.run()
