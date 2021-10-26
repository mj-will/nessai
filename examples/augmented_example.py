#!/usr/bin/env python

# Example of using AugmentedFlowProposal for a multimodel model

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = './outdir/augmented_example/'
logger = setup_logger(output=output, log_level='INFO')

# Define the model


class MultimodalModel(Model):
    """
    A multimodal gaussian model.
    """
    def __init__(self, dims=2):
        self.names = [str(i) for i in range(dims)]
        self.bounds = {i: [-10, 10] for i in self.names}

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        log_p = 0.
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p += (np.log((x[n] >= self.bounds[n][0])
                             & (x[n] <= self.bounds[n][1]))
                      - np.log(self.bounds[n][1] - self.bounds[n][0]))
        return log_p

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point.
        """
        log_l = 0
        # Use a Gaussian logpdf and iterate through the parameters
        for pn in self.names:
            log_l += np.log(norm(-5).pdf(x[pn]) + norm(5).pdf(x[pn]))
        return log_l


flow_config = dict(
        batch_size=1000,
        max_epochs=200,
        patience=50,
        model_config=dict(n_blocks=4, n_neurons=4, n_layers=2, ftype='realnvp',
                          kwargs=dict(batch_norm_between_layers=False,
                                      linear_transform='lu'))
        )

# We `augment_dims` which help the sampler bridge disconnected regions of
# probability. There additional parameters are Gaussian and included in the
# parameters transformed by the flow.
fp = FlowSampler(MultimodalModel(4), output=output, flow_config=flow_config,
                 resume=False, seed=1234, augment_dims=2, nlive=2000,
                 generate_augment='gaussian', expansion_fraction=2.0,
                 flow_class='AugmentedFlowProposal', maximum_uninformed=4000,
                 poolsize=2000)

# And go!
fp.run()
