#!/usr/bin/env python

from scipy.stats import norm, halfnorm
from scipy.special import xlogy

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

output = './outdir/half_gaussian/'
logger = setup_logger(output=output, log_level='DEBUG')


class HalfGaussian(Model):
    """
    Two-dimensional Guassian with a bound at y=0
    """
    def __init__(self):

        self.names = ['x', 'y']
        self.bounds = {'x': [-10, 10], 'y': [0, 10]}

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
        Log likelihood that accepts structured arrays of length
        one with fields:
        x, y, logP, logL
        """
        log_l = 0
        log_l += norm.logpdf(x['x'])
        log_l += halfnorm.logpdf(x['y'])
        return log_l


# Configure the flow
flow_config = dict(
        max_epochs=50,
        patience=10,
        model_config=dict(n_blocks=2, n_neurons=2, n_layers=2)
        )

# Enable boundary inversion for 'y' since we expect the posterior to rail
# against the bounds
fp = FlowSampler(HalfGaussian(), output=output, resume=False, nlive=1000,
                 plot=True, flow_config=flow_config, training_frequency=None,
                 maximum_uninformed=1000, rescale_parameters=True, seed=1234,
                 boundary_inversion=['y'], detect_edges=True)

# And go!
fp.run()
