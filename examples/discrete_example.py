#!/usr/bin/env python

# Example of using nessai with discrete variables
import os

import numpy as np
from scipy.stats import chi

from nessai.flowsampler import FlowSampler
from nessai.livepoint import dict_to_live_points
from nessai.model import Model
from nessai.plot import corner_plot
from nessai.utils import setup_logger

output = "./outdir/discrete_example/"
logger = setup_logger(output=output)


class DiscreteModel(Model):
    def __init__(self):

        self.names = ["mu", "scale", "loc"]
        self.bounds = {"mu": [1, 10], "scale": [0.0, 5.0], "loc": [0.0, 10.0]}

        self.truth = {"mu": 2, "loc": 1.0, "scale": 2.0}

        self.true_dist = chi(
            self.truth["mu"], loc=self.truth["loc"], scale=self.truth["scale"]
        )
        self.data = self.true_dist.rvs(size=100)
        self.vectorised_likelihood = False
        self.verify_with_new_point = True

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype=float) + np.log(
            ~np.mod(x["mu"], 1).astype(bool), dtype=float
        )

    def new_point(self, N=1):
        return dict_to_live_points(
            {
                "mu": np.random.randint(
                    self.bounds["mu"][0], self.bounds["mu"][1] + 1, size=N
                ),
                "loc": np.random.uniform(*self.bounds["loc"], size=N),
                "scale": np.random.uniform(*self.bounds["scale"], size=N),
            }
        )

    def new_point_log_prob(self, x):
        return np.zeros(x.size)

    def log_likelihood(self, x):
        return (
            chi(x["mu"], loc=x["loc"], scale=x["scale"])
            .logpdf(self.data)
            .sum()
        )


model = DiscreteModel()

fs = FlowSampler(
    model,
    output=output,
    nlive=1000,
    reset_flow=4,
    proposal_plots=True,
    reparameterisations={
        "mu": {"reparameterisation": "dequantise", "include_logit": True},
        "scale": "z-score",
        "loc": "z-score",
    },
)
fs.run(plot_posterior=False)

corner_plot(
    fs.posterior_samples,
    truth=model.truth,
    include=model.names,
    filename=os.path.join(output, "corner.png"),
)
