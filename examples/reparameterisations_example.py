#!/usr/bin/env python

# Example of using nessai with `reparameterisations` dictionary. This example
# uses the sample model as the half_gaussian example.

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = './outdir/reparemterisations_example/'
logger = setup_logger(output=output, log_level='INFO')


class HalfGaussianModel(Model):
    """
    A simple two-dimensional Guassian likelihood with a cut at x=0.
    """
    def __init__(self):
        # Names of parameters to sample
        self.names = ['x', 'y']
        # Prior bounds for each parameter
        self.bounds = {'x': [0, 10], 'y': [-10, 10]}

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


# Configure the normalising flow
flow_config = dict(
        max_epochs=50,
        patience=10,
        model_config=dict(n_blocks=2, n_neurons=4, n_layers=1,
                          kwargs=dict(batch_norm_between_layers=True))
        )

# In this example we use the reparameterisation options to specifiy how each
# parameter should be rescaled.
fp = FlowSampler(HalfGaussianModel(), output=output, flow_config=flow_config,
                 resume=False, expansion_fraction=1.0,
                 reparameterisations={
                     'x': {'reparameterisation': 'inversion',
                           'detect_edges': True},
                     'y': 'default'
                }
)

fp.run()
