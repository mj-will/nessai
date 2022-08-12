#!/usr/bin/env python

# Example of using nessai

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

# Setup the logger - credit to the Bilby team for this neat function!
# see: https://git.ligo.org/lscsoft/bilby

output = "./outdir/2d_gaussian_example/"
logger = setup_logger(output=output)

# Define the model, in this case we use a simple 2D gaussian
# The model must contain names for each of the parameters and their bounds
# as a dictionary with arrays/lists with the min and max

# The main functions in the model should be the log_prior and log_likelihood
# The log prior must be able to accept structured arrays of points
# where each field is one of the names in the model and there are two
# extra fields which are `logP` and `logL'


class GaussianModel(Model):
    """A simple two-dimensional Gaussian likelihood."""

    def __init__(self):
        # Names of parameters to sample
        self.names = ["x", "y"]
        # Prior bounds for each parameter
        self.bounds = {"x": [-10, 10], "y": [-10, 10]}

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """
        log_l = np.zeros(x.size)
        # Use a Gaussian logpdf and iterate through the parameters
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l


# The FlowSampler object is used to managed the sampling. Keyword arguments
# are passed to the nested sampling.
fs = FlowSampler(GaussianModel(), output=output, resume=False, seed=1234)

# And go!
fs.run()
