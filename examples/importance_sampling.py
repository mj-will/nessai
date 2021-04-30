#!/usr/bin/env python

# Example of using importance sampling in nessai

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = './outdir/importance_sampling/'
logger = setup_logger(output=output, log_level='INFO')


class GaussianModel(Model):
    """
    A simple two-dimensional Guassian likelihood
    """
    def __init__(self, dims=2):
        # Names of parameters to sample
        self.names = [str(i) for i in range(dims)]
        # Prior bounds for each parameter
        self.bounds = {i: [-10, 10] for i in self.names}

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniforn
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
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """
        log_l = 0
        # Use a Guassian logpdf and iterate through the parameters
        for pn in self.names:
            log_l += norm.logpdf(x[pn])
        return log_l


flow_config = dict(
        batch_size=1000,
        max_epochs=100,
        patience=20,
        model_config=dict(n_blocks=2, n_neurons=2, n_layers=1,
                          kwargs=dict(batch_norm_between_layers=True))
        )

fp = FlowSampler(GaussianModel(2), output=output, flow_config=flow_config,
                 resume=False, seed=1234, nlive=2000,
                 sampling_method='importance_sampling',
                 maximum_uninformed=2000)

fp.run()
