#!/usr/bin/env python

# Example of using Nessai with Bilby (Requires separate installation)
# with unbounded priors.

import bilby
import numpy as np

outdir = "./outdir/"
label = "bilby_unbounded_priors"

# Setup the bilby logger
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Define a likelihood using Bilby


class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self):
        """A very simple Gaussian likelihood"""
        super().__init__(parameters={"x": None, "y": None})

    def log_likelihood(self):
        """Log-likelihood."""
        return -0.5 * (
            self.parameters["x"] ** 2.0 + self.parameters["y"] ** 2.0
        ) - np.log(2.0 * np.pi)


# Define priors, we'll use Gaussians since they're unbounded.
priors = dict(
    x=bilby.core.prior.Gaussian(0, 5, "x"),
    y=bilby.core.prior.Gaussian(0, 10, "y"),
)

# Instantiate the likelihood
likelihood = SimpleGaussianLikelihood()

# Run using bilby.run_sampler, any kwargs are parsed to the sampler
# NOTE: when using Bilby if the priors can be sampled analytically  the flag
# `analytic_priors` enables faster initial sampling. See 'further details' in
# the documentation for more details
# We need to disable the rescaling since we can't rescale without bounds.
# Alternatively a different reparameterisation could be used, in this case
# we can used the 'Rescale' reparameterisation that rescales the inputs by
# a constant, this is useful when we do not have prior bounds we can use.
result = bilby.run_sampler(
    outdir=outdir,
    label=label,
    resume=False,
    plot=True,
    likelihood=likelihood,
    priors=priors,
    sampler="nessai",
    injection_parameters={"x": 0.0, "y": 0.0},
    reparameterisations={
        "scale": {"parameters": ["x", "y"], "scale": [5, 10]}
    },
    analytic_priors=True,
    seed=1234,
)
