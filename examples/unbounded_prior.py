#!/usr/bin/env python

# Example of using nessai with an unbounded prior.

import numpy as np
from scipy.stats import multivariate_normal, norm

from nessai.flowsampler import FlowSampler
from nessai.livepoint import dict_to_live_points
from nessai.model import Model
from nessai.utils import configure_logger

output = "./outdir/unbounded_prior/"
logger = configure_logger(output=output)
rng = np.random.default_rng(1234)

# We define the model as usual but also need to redefine `new_point` since by
# default this tries to draw from within the prior bounds which will fail if
# the bounds are +/- inf.
# We also need to redefine `new_point_log_prob` since this is used to calculate
# the log-probability of the new points.


class GaussianModel(Model):
    """A simple two-dimensional Gaussian Model.

    The prior is Uniform[-10, 10] on x and a Gaussian on y.
    """

    def __init__(self):
        # Names of parameters to sample
        self.names = ["x", "y"]
        # Prior bounds for each parameter
        self.bounds = {"x": [-10, 10], "y": [-np.inf, np.inf]}
        # Scipy distribution for the prior on y
        self._prior_y_dist = norm(scale=5)
        # Distribution for the log-likelihood
        # This is a simple Gaussian with mean 0 and unit covariance
        self._likelihood_dist = multivariate_normal(
            mean=np.zeros(len(self.names)),
            cov=np.eye(len(self.names)),
        )

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
        log_p += self._prior_y_dist.logpdf(x["y"])
        return log_p

    def new_point(self, N=1):
        """Draw n points.

        This is used for the initial sampling. Points do not need to be drawn
        from the exact prior but the initial sampling will be more efficient
        if they are.
        """
        # There are various ways to create live points in nessai, such as
        # from dictionaries and numpy arrays. See nessai.livepoint for options
        d = {
            "x": rng.uniform(self.bounds["x"][0], self.bounds["x"][1], N),
            "y": self._prior_y_dist.rvs(size=N, random_state=rng),
        }
        return dict_to_live_points(d)

    def new_point_log_prob(self, x):
        """Returns the log-probability for a new point.

        Since we have redefined `new_point` we also need to redefine this
        function.
        """
        return self.log_prior(x)

    def log_likelihood(self, x):
        """Returns the log-likelihood.

        In this example we use a simple Gaussian likelihood.
        """
        # Unstructured view returns a view of the inputs as a "normal" numpy
        # array without any fields. It will included the parameters listed in
        # names and in the same order. This is faster then converting from
        # live points to an array.
        x = self.unstructured_view(x)
        return self._likelihood_dist.logpdf(x)


# The sampler can then be configured as usual
fs = FlowSampler(
    GaussianModel(),
    output=output,
    resume=False,
    seed=1234,
)

# And go!
fs.run()
