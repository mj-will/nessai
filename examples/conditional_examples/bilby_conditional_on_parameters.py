#!/usr/bin/env python

# Example of using ConditionalFlowProposal via bilby.

import bilby
import numpy as np

outdir = './outdir/'
label = 'bilby_conditional_on_parameters'

# Setup the bilby logger
bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level='INFO')


class Likelihood(bilby.Likelihood):

    def __init__(self):
        """
        A very simple Gaussian likelihood which ignores the z parameter.
        """
        super().__init__(parameters={'x': None, 'y': None, 'z': None})

    def log_likelihood(self):
        x = self.parameters['x']
        y = self.parameters['y']
        return -0.5*(x ** 2. + y ** 2.) - np.log(2.0 * np.pi)


priors = dict(
    x=bilby.core.prior.Uniform(-10, 10, 'x'),
    y=bilby.core.prior.Uniform(-10, 10, 'y'),
    z=bilby.core.prior.Uniform(-1, 1, 'z')
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

result = bilby.run_sampler(
    outdir=outdir,
    label=label,
    resume=False,
    plot=True,
    likelihood=Likelihood(),
    priors=priors,
    injection_parameters={'x': 0.0, 'y': 0.0},
    sampler='nessai',
    flow_class='ConditionalFlowProposal',
    nlive=2000,
    maximum_uninformed=2000,
    flow_config=flow_config,
    rescale_parameters=['x', 'y'],
    analytic_priors=True,
    prior_parameters=['z'],
    seed=1234
)
