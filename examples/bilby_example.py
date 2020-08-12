#!/usr/bin/env python

# Example of using FlowProposal with Bilby (Requires seperate installation)
# See 2d_gaussian.py for a more detailed explanation

import bilby
import numpy as np
import torch


torch.set_num_threads(1)

# The output from the sampler will be saved to:
# '$outdir/$label_flowproposal/'
# alongside the usual bilby outputs
outdir = './outdir/'
label = 'bilby_example'

bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level='DEBUG')

# Define a likelihood using Bilby


class SimpleGaussianLikelihood(bilby.Likelihood):

    def __init__(self):
        """
        A very simple Gaussian likelihood
        """
        super().__init__(parameters={'x': None, 'y': None})

    def log_likelihood(self):
        x = self.parameters['x']
        y = self.parameters['y']
        return -0.5*(x ** 2. + y ** 2.) - np.log(2.0 * np.pi)


# Define priors (this provides the bounds that are then used in FlowProposal)
priors = dict(x=bilby.core.prior.Uniform(-10, 10, 'x'),
              y=bilby.core.prior.Uniform(-10, 10, 'y'))

# Instantiate the likleihood
likelihood = SimpleGaussianLikelihood()

# Configure the normalisng flow
flow_config = dict(
        max_epochs=50,
        patience=10,
        model_config=dict(n_blocks=4, n_neurons=8, n_layers=2,
                          device_tag='cpu',
                          kwargs=dict(batch_norm_between_layers=True))
        )

# Run using bilby.run_sampler, any kwargs are parsed to the sampler
# NOTE: when using Bilby if the priors can be sampled analytically  the flat
# `analytic_priors` enables faster initial sampling
# `proposal_plots` enables plots for each block of training and each
# population stage
result = bilby.run_sampler(outdir=outdir, label=label, resume=False, plot=True,
                           likelihood=likelihood, priors=priors,
                           sampler='flowproposal', nlive=1000,
                           maximum_uninformed=1000, flow_config=flow_config,
                           rescale_parameters=True,
                           injection_parameters={'x': 0.0, 'y': 0.0},
                           proposal_plots=True, analytic_priors=True,
                           training_frequency=1000, seed=1234)
