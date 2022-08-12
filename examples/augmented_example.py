#!/usr/bin/env python

# Example of using AugmentedFlowProposal for a multimodal model
# Note that this feature is experimental and not fully tested.

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = "./outdir/augmented_example/"
logger = setup_logger(output=output)


class MultimodalModel(Model):
    """A multimodal gaussian model."""

    def __init__(self, dims=2):
        self.names = [str(i) for i in range(dims)]
        self.bounds = {i: [-10, 10] for i in self.names}

    def log_prior(self, x):
        """Log-prior"""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        """Log likelihood."""
        log_l = np.zeros(x.size)
        for n in self.names:
            log_l += np.log(norm(-5).pdf(x[n]) + norm(5).pdf(x[n]))
        return log_l


# We add `augment_dims` which help the sampler bridge disconnected regions of
# probability. There additional parameters are Gaussian and included in the
# parameters transformed by the flow.
fs = FlowSampler(
    MultimodalModel(4),
    output=output,
    resume=False,
    seed=1234,
    augment_dims=2,
    flow_class="AugmentedFlowProposal",
)

# And go!
fs.run()
