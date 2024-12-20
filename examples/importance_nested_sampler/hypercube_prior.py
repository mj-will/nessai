#!/usr/bin/env python

# Example of using the importance nested sampler with a non-uniform
# prior in the unit hypercube space.

import os

import numpy as np
from scipy.stats import norm, truncnorm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.plot import corner_plot
from nessai.utils import configure_logger

output = os.path.join("outdir", "ins_non_uniform_prior")
logger = configure_logger(output=output)


class ModelWithNonUniformPrior(Model):
    """A likelihood that has a non-uniform prior in the unit-hypercube"""

    def __init__(self, dims):
        self.names = [f"x_{d}" for d in range(dims)]
        self.bounds = {n: [-10.0, 10.0] for n in self.names}

        # Gaussian prior truncated on [-10, 10] with mean 0 and scale 0.5
        scale = 0.5
        self.prior_dist = truncnorm(-10 / scale, 10 / scale, scale=scale)

        # Define the distribution in the uniform hypercube
        # Will be centred at 0.5
        loc = 0.5
        h_scale = scale / 20
        self.hypercube_prior_dist = truncnorm(
            (0 - loc) / h_scale,  # Limits [0, 1]
            (1 - loc) / h_scale,
            loc=loc,
            scale=h_scale,
        )
        self.likelihood_dist = norm(loc=1.0, scale=0.5)

    def log_prior(self, x):
        """Log-prior"""
        log_p = np.log(self.in_bounds(x), dtype=float)
        log_p += self.prior_dist.logpdf(self.unstructured_view(x)).sum(axis=-1)
        return log_p

    def log_likelihood(self, x):
        "Log-likelihood"
        return self.likelihood_dist.logpdf(self.unstructured_view(x)).sum(
            axis=-1
        )

    def from_unit_hypercube(self, x):
        """Mapping from the unit hypercube.

        This mapping does not map from uniform to the correct prior.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out

    def log_prior_unit_hypercube(self, x) -> np.ndarray:
        """The log-prior defined in the unit hypercube space.

        This must be defined based on the how `from_unit_hypercube` is
        defined.
        """
        return np.log(
            self.in_unit_hypercube(x), dtype=float
        ) + self.hypercube_prior_dist.logpdf(self.unstructured_view(x)).sum(
            axis=-1
        )


# Run standard nessai for reference
model = ModelWithNonUniformPrior(2)
fs = FlowSampler(
    model,
    nlive=1000,
    output=os.path.join(output, "standard"),
    resume=False,
    seed=1234,
    importance_nested_sampler=False,
)
fs.run()

# Run the importance nested sampler
model = ModelWithNonUniformPrior(2)
fs_ins = FlowSampler(
    model,
    nlive=1000,
    output=os.path.join(output, "ins"),
    resume=False,
    seed=1234,
    importance_nested_sampler=True,
)
fs_ins.run()

# Compare the evidences
print(f"Log-evidences: {fs.log_evidence:.3f} vs {fs_ins.log_evidence:.3f}")

# Plot a comparison of the posteriors
fig = corner_plot(fs.posterior_samples, color="C0", include=model.names)
fig = corner_plot(
    fs_ins.posterior_samples,
    color="C1",
    fig=fig,
    include=model.names,
    filename=os.path.join(output, "comparison.png"),
)
