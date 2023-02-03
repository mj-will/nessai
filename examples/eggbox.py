#!/usr/bin/env python

# Example of using nessai with the eggbox function. Nessai is not well
# suited to this sorted problem. The augmented proposal method is used to
# to improve the results but ultimately likelihoods that this multi-modal
# are a challenge.

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger


output = "./outdir/eggbox/"
logger = setup_logger(output=output)


class EggboxModel(Model):
    """
    Eggbox problem from https://arxiv.org/pdf/0809.3437v1.pdf.

    Based on the example in cpnest: \
        https://github.com/johnveitch/cpnest/blob/master/examples/eggbox.py
    """

    def __init__(self, dims):
        self.names = [str(i) for i in range(dims)]
        self.bounds = {n: [0, 10 * np.pi] for n in self.names}

    def log_prior(self, x):
        """Log-prior."""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood."""
        return (
            np.prod(np.cos(self.unstructured_view(x) / 2.0), axis=-1) + 2
        ) ** 5.0


flow_config = dict(
    patience=50,  # Make sure the flow trains for longer
    model_config=dict(n_blocks=6, n_neurons=8),
)

# Sampling the Eggbox will be inefficient, so we increase the maximum size of
# the pool of proposal points. We also allow the initial sampling without
# the flow to run until it becomes inefficient rather than a fixed iteration.
fs = FlowSampler(
    EggboxModel(2),
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=1234,
    flow_class="AugmentedFlowProposal",
    augment_dims=2,  # Add two augment parameters
    maximum_uninformed="inf",  # Run initial sampling until it's inefficient
    max_poolsize_scale=50,  # Increase the maximum poolsize
    max_threads=3,
    n_pool=2,
)

fs.run()
