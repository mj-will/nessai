#!/usr/bin/env python

# Example of using the importance nested sampler in nessai

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.livepoint import live_points_to_array

output = './outdir/basic_importance_example/'
logger = setup_logger(output=output)

# Define the model. For the importance sampler we must define mappings to
# and from the unit hyper-cube.


class RosenbrockModel(Model):
    """
    A Rosenbrock likelihood.
    """
    def __init__(self, dims):
        self.names = [f'x_{d}' for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        """Log-prior"""
        log_p = np.log(self.in_bounds(x))
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood"""
        x = live_points_to_array(x, self.names)
        x = np.atleast_2d(x)
        return - (np.sum(
            100. * (x[:, 1:] - x[:, :-1] ** 2.) ** 2.
            + (1. - x[:, :-1]) ** 2.,
            axis=1
        ))

    def to_unit_hypercube(self, x):
        """Map to the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (
                (x[n] - self.bounds[n][0])
                / (self.bounds[n][1] - self.bounds[n][0])
            )
        return x_out

    def from_unit_hypercube(self, x):
        """Map from the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (
                (self.bounds[n][1] - self.bounds[n][0])
                * x[n] + self.bounds[n][0]
            )
        return x_out


# The FlowSampler object is used to managed the sampling as has more
# configuration options
fs = FlowSampler(
    RosenbrockModel(4),
    nlive=5000,
    output=output,
    resume=False,
    seed=1234,
    importance_sampler=True,
)

# And go!
fs.run()
