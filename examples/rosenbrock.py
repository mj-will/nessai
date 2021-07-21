#!/usr/bin/env python

# Example of using nessai for a Rosenbrock likelihood

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
from nessai.livepoint import live_points_to_array

output = './outdir/rosenbrock/'
logger = setup_logger(output=output, log_level='WARNING')


class RosenbrockModel(Model):
    """Rosenbrock function defined in n dimensions on [-5, 5]^n.

    Based on the example in cpnest:\
        https://github.com/johnveitch/cpnest/blob/master/examples/rosenbrock.py
    """
    def __init__(self, dims):
        self.names = [f'x_{d}' for d in range(dims)]
        self.bounds = {n: [-5.0, 5.0] for n in self.names}

    def log_prior(self, x):
        """Uniform prior"""
        log_p = np.log(self.in_bounds(x))
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        """Log-likelihood"""
        x = live_points_to_array(x, self.names)[np.newaxis, :]
        return -(np.sum(
            100. * (x[:, 1:] - x[:, :-1] ** 2.) ** 2. + (1. - x[:, :-1]) ** 2.,
            axis=1)
        )


model = RosenbrockModel(2)

# Use a more complex flow compared to simpler examples.
flow_config = dict(
    batch_size=1000,
    max_epochs=200,
    patience=20,
    model_config=dict(n_blocks=4, n_neurons=4, n_layers=1)
)

# Configure the sampler.
# We use the logit rescaling since parts of the likelihood are close to the
# prior bounds
fs = FlowSampler(
    model,
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=1234,
    nlive=2000,
    poolsize=2000,
    maximum_uninformed=4000,
    proposal_plots=False,
    reparameterisations={'logit': {'parameters': model.names}}
)

# And go!
fs.run()
