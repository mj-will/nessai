#!/usr/bin/env python

"""
Example of using `corner_plot` to plot the posterior distribution with truth
values.

Requires corner.
"""

import os
import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.plot import corner_plot
from nessai.utils import setup_logger


output = "./outdir/corner_plot_example/"
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


fs = FlowSampler(
    GaussianLikelihood(data, bounds),
    output=output,
    resume=False,
    seed=1234,
)

fs.run(plot=False)

# Produce a corner plot that includes the true values for each parameter and
# uses custom labels for each.
corner_plot(
    fs.posterior_samples,
    include=list(truth.keys()),
    truths=list(truth.values()),
    labels=[r"$\mu$", r"$\sigma$"],
    filename=os.path.join(output, "posterior_w_truth.png"),
)
