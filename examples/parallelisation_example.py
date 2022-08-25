#!/usr/bin/env python

"""
Example of parallelising the likelihood evaluation in nessai.

Shows the two methods supported in nessai: setting n_pool or using a
user-defined pool.
"""

import numpy as np
from multiprocessing import Pool

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.utils.multiprocessing import initialise_pool_variables


output = "./outdir/parallelisation_example/"
logger = setup_logger(output=output)


# Generate the data
truth = {"mu": 1.7, "sigma": 0.7}
bounds = {"mu": [-3, 3], "sigma": [0.01, 3]}
n_points = 1000
data = np.random.normal(truth["mu"], truth["sigma"], size=n_points)


class GaussianLikelihood(Model):
    """
    Gaussian likelihood with the mean and standard deviation as the parameters
    to infer.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Array of data.
    bounds : dict
        The prior bounds.
    """

    def __init__(self, data, bounds):
        self.names = list(bounds.keys())
        self.bounds = bounds
        self.data = data

    def log_prior(self, x):
        """Uniform prior on both parameters."""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        """Gaussian likelihood."""
        log_l = np.sum(
            -np.log(x["sigma"])
            - 0.5 * ((self.data - x["mu"]) / x["sigma"]) ** 2
        )
        return log_l


# Using n_pool
logger.warning("Running nessai with n_pool")
# Configure the sampler with 3 total threads, 2 of which are used for
# evaluating the likelihood.
fs = FlowSampler(
    GaussianLikelihood(data, bounds),
    output=output,
    resume=False,
    seed=1234,
    pytorch_threads=2,  # Allow pytorch to use 2 threads
    n_pool=2,  # Threads for evaluating the likelihood
)

# Run the sampler
fs.run()

# Using a user-defined pool
logger.warning("Running nessai with a user-defined pool")

# Must initialise the global variables for the pool prior to starting it
model = GaussianLikelihood(data, bounds)
initialise_pool_variables(model)
# Define the pool
pool = Pool(2)

fs = FlowSampler(
    model,
    output=output,
    resume=False,
    seed=1234,
    pool=pool,  # User-defined pool
)

# Run the sampler
# The pool will automatically be closed. This can be disabled by passing
# `close_pool=False` to the sampler.
fs.run()
