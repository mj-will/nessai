#!/usr/bin/env python

# Example of using the boundary inversion available in nessai.
# See also the reparameterisations example.

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = "./outdir/half_gaussian/"
logger = setup_logger(output=output)


class HalfGaussian(Model):
    """Two-dimensional Gaussian with a bound at y=0."""

    def __init__(self):
        self.names = ["x", "y"]
        self.bounds = {"x": [-10, 10], "y": [0, 10]}

    def log_prior(self, x):
        """Log-prior"""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood"""
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l


# Enable boundary inversion for 'y' since we expect the posterior to rail
# against the bounds
fs = FlowSampler(
    HalfGaussian(),
    output=output,
    resume=False,
    seed=1234,
    boundary_inversion=["y"],
)

# And go!
fs.run()
