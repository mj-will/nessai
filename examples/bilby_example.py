#!/usr/bin/env python

# Example of using nessai with Bilby (Requires separate installation)
# See 2d_gaussian.py for a more detailed explanation of using nessai

import bilby
import numpy as np

# The output from the sampler will be saved to:
# '$outdir/$label_nessai/'
# alongside the usual bilby outputs
outdir = "./outdir/"
label = "bilby_example"

# Setup the bilby logger this will also configure the nessai logger.
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


# Define priors (this provides the bounds that are then used in nessai)
priors = dict(
    x=bilby.core.prior.Uniform(-10, 10, "x"),
    y=bilby.core.prior.Uniform(-10, 10, "y"),
)

# Instantiate the likelihood
likelihood = SimpleGaussianLikelihood()

# Run using bilby.run_sampler, any kwargs are parsed to the sampler
# NOTE: when using Bilby if the priors can be sampled analytically  the flag
# `analytic_priors` enables faster initial sampling. See 'further details' in
# the documentation for more details
result = bilby.run_sampler(
    outdir=outdir,
    label=label,
    resume=False,
    plot=True,
    likelihood=likelihood,
    priors=priors,
    sampler="nessai",
    injection_parameters={"x": 0.0, "y": 0.0},
    analytic_priors=True,
    seed=1234,
)
