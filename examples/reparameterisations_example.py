#!/usr/bin/env python

# Example of using nessai with `reparameterisations` dictionary. This example
# uses the same model as the half_gaussian example.

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = "./outdir/reparameterisations_example/"
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


# In this example we use the reparameterisation options to specify how each
# parameter should be rescaled.
# In this case we'll tell nessai to use an inversion for y, but only at the
# lower bound.
fs = FlowSampler(
    HalfGaussian(),
    output=output,
    resume=False,
    seed=1234,
    reparameterisations={
        "x": "default",
        "y": {
            "reparameterisation": "inversion",
            "detect_edges_kwargs": {"allowed_bounds": ["lower"]},
        },
    },
)

fs.run()
