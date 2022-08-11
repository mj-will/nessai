#!/usr/bin/env python

# Example of using nessai for a Rosenbrock likelihood.
# Also shows how to configure the normalising flow.

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = "./outdir/rosenbrock/"
logger = setup_logger(output=output)


class RosenbrockModel(Model):
    """Rosenbrock function defined in n dimensions on [-5, 5]^n.

    Based on the example in cpnest:\
        https://github.com/johnveitch/cpnest/blob/master/examples/rosenbrock.py
    """

    def __init__(self, dims):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        """Uniform prior"""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood"""
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


model = RosenbrockModel(5)

# The Rosenbrock likelihood is more complex, so we configure the normalising
# flow to improve nessai's performance.
flow_config = dict(model_config=dict(n_blocks=4, n_neurons=10, n_layers=3))

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
