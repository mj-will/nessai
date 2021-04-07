#!/usr/bin/env python

# Example of using nessai

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.proposal.augmented import AugmentedFlowProposal

# Setup the logger - credit to the Bilby team for this neat function!
# see: https://git.ligo.org/lscsoft/bilby

output = './outdir/augmented_example_gaussian_4d/'
logger = setup_logger(output=output, log_level='INFO')

# Define the model, in this case we use a simple 2D gaussian
# The model must contain names for each of the parameters and their bounds
# as a dictionary with arrays/lists with the min and max

# The main functions in the model should be the log_prior and log_likelihood
# The log prior must be able to accept structed arrays of points
# where each field is one of the names in the model and there are two
# extra fields which are `logP` and `logL'


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
            log_l += np.log(norm(-5).pdf(x[pn]) + norm(5).pdf(x[pn]))
        return log_l


# The normalsing flow that is trained to produce the proposal points
# is configured with a dictionary that contains the parameters related to
# training (e.g. learning rate (lr)) and model_config for the configuring
# the flow itself (neurons, number of trasformations etc)
flow_config = dict(
        max_epochs=200,
        patience=20,
        model_config=dict(n_blocks=4, n_neurons=8, n_layers=2, ftype='realnvp',
                          kwargs=dict(batch_norm_between_layers=False))
        )

# The FlowSampler object is used to managed the sampling as has more
# configuration options
fp = FlowSampler(GaussianModel(4), output=output, flow_config=flow_config,
                 resume=False, seed=1234, augment_features=2, nlive=2000,
                 generate_augment='gaussian',
                 flow_class=AugmentedFlowProposal, maximum_uninformed=2000)

# And go!
fp.run()
