#!/usr/bin/env python

# Example of using nessai

import numpy as np
from scipy.stats import norm

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = './outdir/conditional_on_parameters/'
logger = setup_logger(output=output, log_level='DEBUG')


class PriorParametersModel(Model):

    names = ['x', 'y', 'c1', 'c2']
    # dictionary of bounds bounds
    bounds = {'x': [-10, 10], 'y': [-10, 10], 'c1': [-1, 1], 'c2': [-1, 1]}

    def log_prior(self, x):
        """
        Log prior that accepts structured arrays with fields:
        x, y, logP, logL
        """
        log_p = np.log(self.in_bounds(x))
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """
        Log likelihood that does not include z.
        """
        log_l = 0
        log_l += norm.logpdf(x['x'])
        log_l += norm.logpdf(x['y'])
        return log_l

    def sample_parameter(self, name, n=1):
        """
        Return samples for a specific parameter.
        """
        return np.random.uniform(self.bounds[name][0], self.bounds[name][1], n)

    def parameter_log_prior(self, x, name):
        """
        Method that computes the prior for a single parameter.
        """
        return (
            np.log(self.parameter_in_bounds(x, name)) -
            np.log(self.bounds[name][1] - self.bounds[name][0])
        )


flow_config = dict(
    lr=0.001,
    batch_size=1000,
    max_epochs=500,
    patience=50,
    model_config=dict(
        n_blocks=2,
        n_neurons=4,
        n_layers=2,
    )
)

fp = FlowSampler(
    PriorParametersModel(),
    output=output,
    resume=False,
    nlive=2000,
    maximum_uninformed=2000,
    rescale_parameters=['x', 'y'],
    update_bounds=True,
    seed=1234,
    plot=True,
    flow_class='ConditionalFlowProposal',
    flow_config=flow_config,
    prior_parameters=['c1', 'c2'],
)

# And go!
fp.run(plot=True)
