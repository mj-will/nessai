#!/usr/bin/env python

# Example of parallelising the likelihood evaluation in nessai

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger


output = './outdir/parallelisation_example/'
logger = setup_logger(output=output, log_level='WARNING')


class GaussianLikelihood(Model):
    """
    Gaussian likelihood with the mean and standard deviation as the parameters
    to infer.

    Parameters
    ----------
    n_points : int, optional
        Number of points to sample for the data. More points will lead to
        a slower likelihood.
    """
    def __init__(self, n_points=1000):
        self.names = ['mu', 'sigma']
        self.bounds = {'mu': [-3, 3], 'sigma': [0.01, 3]}
        self.truth = {'mu': 1.7, 'sigma': 0.7}

        self.data = np.random.normal(
            self.truth['mu'], self.truth['sigma'], size=n_points
        )

    def log_prior(self, x):
        """
        Uniform prior on both parameters.
        """
        log_p = 0.
        for n in self.names:
            log_p += (np.log((x[n] >= self.bounds[n][0])
                             & (x[n] <= self.bounds[n][1]))
                      - np.log(self.bounds[n][1] - self.bounds[n][0]))
        return log_p

    def log_likelihood(self, x):
        """
        Gaussian likelihood.
        """
        log_l = np.sum(
            - np.log(x['sigma']) -
            0.5 * ((self.data - x['mu']) / x['sigma']) ** 2
        )
        return log_l


flow_config = dict(
    batch_size=1000,
    max_epochs=200,
    patience=20,
    model_config=dict(n_blocks=2, n_neurons=4, n_layers=2)
)

# Configure the sampler with 3 total threads, 2 of which are used for
# evaluating the likelihood.
fs = FlowSampler(
    GaussianLikelihood(),
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=1234,
    max_threads=3,               # Maximum number of threads
    n_pool=2,                    # Threads for evaluating the likelihood
    nlive=2000,
    maximum_uninformed=2000,
    proposal_plots=False,
)

# Run the sampler
fs.run()
