#!/usr/bin/env python

# Example of using nessai

from scipy.stats import norm
from scipy.special import xlogy
import torch

from nessai.flowsampler import FlowSampler
from nessai.proposal import ConditionalFlowProposal
from nessai.model import Model
from nessai.utils import setup_logger

# This prevents torch from using all of the available threads when
# running on the CPU
torch.set_num_threads(1)

# Setup the logger - credit to the Bilby team for this neat function!
# see: https://git.ligo.org/lscsoft/bilby

output = './outdir/conditional_flow/'
logger = setup_logger(output=output, log_level='DEBUG')

# Define the model, in this case we use a simple 2D gaussian
# The model must contain names for each of the parameters and their bounds
# as a dictionary with arrays/lists with the min and max

# The main functions in the model should be the log_prior and
# log_likelihood
# The log prior must be able to accept structed arrays of points
# where each field is one of the names in the model and there are two
# exra field which are `logP` and `logL'


class Gaussian(Model):

    names = ['x', 'y', 'z']
    # dictionary of bounds bounds
    bounds = {'x': [-10, 10], 'y': [-10, 10], 'z': [-1, 1]}

    def log_prior(self, x):
        """
        Log prior that accepts structured arrays with fields:
        x, y, logP, logL
        """
        log_p = 0.
        for i, n in enumerate(self.names):
            log_p += xlogy(1, (x[n] >= self.bounds[n][0])
                           & (x[n] <= self.bounds[n][1]))
            - xlogy(1, self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """
        Log likelihood that accepts structured arrays of length one
        with fields:

        x, y, logP, logL
        """
        log_l = 0
        log_l += norm.logpdf(x['x'] - x['z'])
        log_l += norm.logpdf(x['y'] - x['z'])
        return log_l


# The normalsing flow that is trained to produce the proposal points
# is configured with a dictionary that contains the parameters related to
# training (e.g. learning rate (lr)) and model_config for the configuring
# the flow itself (neurons, number of trasformations etc)
flow_config = dict(
        max_epochs=50,
        patience=10,
        model_config=dict(n_blocks=2, n_neurons=4, n_layers=1,
                          device_tag='cpu',
                          kwargs=dict(batch_norm_between_layers=True))
        )

# The FlowSampler object is used to managed the sampling as has more
# configuration options
fp = FlowSampler(Gaussian(), output=output, resume=False, nlive=1000,
                 plot=True, flow_config=flow_config, training_frequency=None,
                 maximum_uninformed=1000, rescale_parameters=True, seed=1234,
                 proposal_plots=True, uniform_parameters=False,
                 flow_class=ConditionalFlowProposal, poolsize=1000,
                 update_poolsize=True)

# And go!
fp.run(plot=True)
