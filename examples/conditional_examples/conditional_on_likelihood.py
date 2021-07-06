#!/usr/bin/env python

# Example of using nessai

from scipy.stats import norm
from scipy.special import xlogy

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = './outdir/conditional_flow/'
logger = setup_logger(output=output, log_level='DEBUG')


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
        log_l += norm.logpdf(x['x'])
        log_l += norm.logpdf(x['y'])
        return log_l


flow_config = dict(
    lr=0.001,
    batch_size=1000,
    max_epochs=500,
    patience=50,
    model_config=dict(
        n_blocks=4,
        n_neurons=4,
        n_layers=2,
        ftype='spline',
        kwargs=dict(
            batch_norm_between_layers=True,
            num_bins=10,
            base_distribution='uniform'
        )
    )
)

fp = FlowSampler(
    Gaussian(),
    output=output,
    resume=False,
    nlive=2000,
    maximum_uninformed=4000,
    rescale_parameters=True,
    rescale_bounds=[0.0+1e-6, 1.0-1e-6],
    update_bounds=True,
    seed=1234,
    plot=True,
    conditional_likelihood=True,
    flow_class='ConditionalFlowProposal',
    flow_config=flow_config,
    latent_prior='flow',
)

# And go!
fp.run(plot=True)
