#!/usr/bin/env python

# Example of using nessai with an unbounded prior.

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.livepoint import dict_to_live_points
from nessai.model import Model
from nessai.utils import setup_logger

output = "./outdir/unbounded_prior/"
logger = setup_logger(output=output)

# We define the model as usual but also need to redefine `new_point` since by
# default this tries to draw from within the prior bounds which will fail if
# the bounds are +/- inf.


class GaussianModel(Model):
    """A simple two-dimensional Gaussian Model.

    The prior is Uniform[-10, 10] on x and a unit-Gaussian on y.
    """

    def __init__(self):
        # Names of parameters to sample
        self.names = ["x", "y"]
        # Prior bounds for each parameter
        self.bounds = {"x": [-10, 10], "y": [-np.inf, np.inf]}

    def log_prior(self, x):
        """
        Returns the log-prior.

        Checks if the points are in bounds.
        """
        # Check if the values are in the priors bounds, will return -inf if not
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Uniform on x
        log_p -= np.log(self.bounds["x"][1] - self.bounds["x"][0])
        # Gaussian on y
        log_p += norm(scale=5).logpdf(x["y"])
        return log_p

    def new_point(self, N=1):
        """Draw n points.

        This is used for the initial sampling. Points do not need to be drawn
        from the exact prior but algorithm will be more efficient if they are.
        """
        # There are various ways to create live points in nessai, such as
        # from dictionaries and numpy arrays. See nessai.livepoint for options
        d = {
            "x": np.random.uniform(
                self.bounds["x"][0], self.bounds["x"][1], N
            ),
            "y": norm(scale=5).rvs(size=N),
        }
        return dict_to_live_points(d)

    def log_likelihood(self, x):
        """Returns the log-likelihood.

        In this example we use a simple Gaussian likelihood.
        """
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += norm.logpdf(x[n])
        return log_l


# The sampler can then be configure as usual
fs = FlowSampler(
    GaussianModel(),
    output=output,
    resume=False,
    seed=1234,
)

# And go!
fs.run()
