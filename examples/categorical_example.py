#!/usr/bin/env python

# Example of using categorical parameters with nessai.
# Credit to Greg Ashton for proposing this idea with BilbyMCMC

import os
import numpy as np
import pandas as pd
import seaborn as sns

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.livepoint import numpy_array_to_live_points

output = './outdir/categorical_example/both/'
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
    def __init__(self, n_samples=32, noise_scale=0.5, models=[0, 1]):
        self.names = ['m', 'c', 'model']
        self.bounds = {'m': [0, 5], 'c': [-1, 1], 'model': [0, 1]}
        self.truth = {'m': 2.3, 'c': 0.5, 'model': 0}
        self.allowed_models = models

        self.x = np.linspace(0, 1.5, n_samples)
        self.data = \
            self.truth['m'] * self.x + self.truth['c'] + \
            noise_scale * np.random.randn(n_samples)

        self.categorical_parameters = ['model']

    def new_point(self, N=1):
        """Draw new samples.

        Since we have a categorical parameter we need to refine this method,
        the default version trys to sample uniformly between the prior bounds.
        """
        x = numpy_array_to_live_points(np.zeros([N, self.dims]), self.names)
        x['m'] = np.random.uniform(*self.bounds['m'], N)
        x['c'] = np.random.uniform(*self.bounds['c'], N)
        x['model'] = np.random.choice(self.allowed_models, size=N)
        return x

    def new_point_log_prob(self, x):
        """Probability for the `new_point` method."""
        return self.log_prior(x)

    def log_prior(self, x):
        """Returns log of prior given a live point.

        We assume a uniform prior on m, c and the model parameter.
        """
        return -(np.log(5) + np.log(2) + np.log(len(self.allowed_models))) * \
            np.ones(x.size)

    def log_likelihood(self, x):
        """Returns log likelihood of given live point.

        This is where we define the two models.
        """
        if x['model'] == 0:
            # Linear model
            line = x['m'] * self.x + x['c']
        elif x['model'] == 1:
            # Non-linear model
            line = x['m'] * (self.x ** 0.5) + x['c']
        else:
            raise RuntimeError(f"Invalid value for w: {x['model']}")

        log_l = -0.5 * np.sum((line - self.data) ** 2)
        return log_l


flow_config = dict(
        batch_size=1000,
        max_epochs=200,
        patience=20,
        model_config=dict(n_blocks=4, n_neurons=4, n_layers=2)
        )

# Can change this to run inferences on a single model.
model = CategoricalModel(models=[0, 1])

# Configure the sampler
# Make sure we're using 'ConditionalFlowProposal'
fp = FlowSampler(
    model,
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=12345,
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

# Posterior samples
posterior_samples = fp.posterior_samples
# Ratio of samples in the posterior
w = posterior_samples['model']
m0 = (w == 0).sum()
m1 = (w == 1).sum()
logger.info('Number of samples in posterior:')
logger.info(f'Model 0: {m0}')
logger.info(f'Model 1: {m1}')
logger.info(f'Ratio: {m0 / m1:.3f}')
# Plot posteriors using Seaborn and hue argument.
df = pd.DataFrame(posterior_samples)
df = df.replace(to_replace={'model': {0.0: 'Linear', 1.0: 'Exponential'}})
fig = sns.jointplot(
    data=df,
    x='m',
    y='c',
    hue='model',
    kind='kde'
)
truth_colour = sns.color_palette()[2]
fig.ax_joint.axvline(model.truth['m'], c=truth_colour)
fig.ax_joint.axhline(model.truth['c'], c=truth_colour)
fig.ax_marg_x.axvline(model.truth['m'], c=truth_colour)
fig.ax_marg_y.axhline(model.truth['c'], c=truth_colour)
fig.savefig(os.path.join(output, 'individual_posteriors.png'))
