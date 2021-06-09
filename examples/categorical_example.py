#!/usr/bin/env python

# Example of using categorical parameters with nessai.
# Credit to Greg Ashton for proposing this idea with BilbyMCMC

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.livepoint import numpy_array_to_live_points

output = './outdir/categorical_example/'
logger = setup_logger(output=output, log_level='INFO')


class CategoricalModel(Model):
    """Model with categorical parameters.

    Data is modelled as line with Gaussian noise

    There are two possible versions of the likelihood:
    0. Linear model mx + c
    1. Non-linear model mx^0.5 + c
    A weight parameter `w` allows the sampler to move between likelihoods
    and this is set as a categorical parameter in the model.
    """
    def __init__(self, n_samples=32, noise_scale=0.5):
        # Names of parameters to sample
        self.names = ['m', 'c', 'w']
        # Prior bounds for each parameter
        self.bounds = {'m': [0, 5], 'c': [-1, 1], 'w': [0, 1]}
        self.truth = {'m': 2.3, 'c': 0.5, 'w': 0}

        self.x = np.linspace(0, 2, n_samples)
        self.data = \
            self.truth['m'] * self.x + self.truth['c'] + \
            noise_scale * np.random.randn(n_samples)

        self.categorical_parameters = ['w']

    def new_point(self, N=1):
        """Draw new samples.

        Since we have a categorical parameter we need to refine this method,
        the default version trys to sample uniformly between the prior bounds.
        """
        x = numpy_array_to_live_points(np.zeros([N, self.dims]), self.names)
        x['m'] = np.random.uniform(*self.bounds['m'], N)
        x['c'] = np.random.uniform(*self.bounds['c'], N)
        x['w'] = np.random.choice([0, 1], size=N)
        return x

    def new_point_log_prob(self, x):
        return self.log_prior(x)

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniforn
        priors on each parameter.
        """
        return -(np.log(5) + np.log(2) + np.log(2)) * np.ones(x.size)

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """
        if x['w'] == 0:
            # Linear model
            line = x['m'] * self.x + x['c']
        elif x['w'] == 1:
            # Non-linear model
            line = x['m'] * (self.x ** 0.5) + x['c']
        else:
            raise RuntimeError(f"Invalid value for w: {x['w']}")

        log_l = -0.5 * np.sum((line - self.data) ** 2)
        return log_l


flow_config = dict(
        batch_size=1000,
        max_epochs=200,
        patience=20,
        model_config=dict(n_blocks=4, n_neurons=4, n_layers=2)
        )

fp = FlowSampler(
    CategoricalModel(),
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=1234,
    flow_class='ConditionalFlowProposal',
    nlive=2000,
    maximum_uninformed=2000,
    poolsize=2000,
    reparameterisations={
        'm': {'reparameterisation': 'default'},
        'c': {'reparameterisation': 'default'}
    }
)

# And go!
fp.run()

# Ratio of samples in the posterior
w = fp.posterior_samples['w']
m0 = (w == 0).sum()
m1 = (w == 1).sum()
logger.info('Number of samples in posterior:')
logger.info(f'Model 0: {m0}')
logger.info(f'Model 1: {m1}')
logger.info(f'Ratio: {m0/m1:.3f}')
