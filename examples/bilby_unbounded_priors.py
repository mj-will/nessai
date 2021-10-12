#!/usr/bin/env python

# Example of using Nessai with Bilby (Requires separate installation)
# with unbounded priors. We use bilby since the priors include a sample
# method which is used to redefine `Model.new_point`. Without this we
# would have to redefine the method ourselves.

import bilby
import numpy as np

# The output from the sampler will be saved to:
# '$(outdir)/$(label)_nessai/'
# alongside the usual bilby outputs
outdir = './outdir/'
label = 'bilby_unbounded_priors'

# Setup the bilby logger
bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level='INFO')

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


# Define priors, we'll use Gaussians since they're unbounded.
priors = dict(x=bilby.core.prior.Gaussian(0, 5, 'x'),
              y=bilby.core.prior.Gaussian(0, 10, 'y'))

# Instantiate the likelihood
likelihood = SimpleGaussianLikelihood()

# Configure the normalising flow
flow_config = dict(
        max_epochs=50,
        patience=10,
        model_config=dict(n_blocks=2, n_neurons=4, n_layers=2,
                          device_tag='cpu',
                          kwargs=dict(batch_norm_between_layers=True))
        )

# Run using bilby.run_sampler, any kwargs are parsed to the sampler
# NOTE: when using Bilby if the priors can be sampled analytically  the flag
# `analytic_priors` enables faster initial sampling. See 'further details' in
# the documentation for more details
# We need to disable the rescaling since we can't rescale without bounds.
# Alternatively a different reparameterisation could be used, in this case
# we can used the 'Rescale' reparameterisation that rescales the inputs by
# a constant, this is useful when we do not prior bounds we can use.
result = bilby.run_sampler(outdir=outdir, label=label, resume=False, plot=True,
                           likelihood=likelihood, priors=priors,
                           sampler='nessai', nlive=1000,
                           maximum_uninformed=2000, flow_config=flow_config,
                           injection_parameters={'x': 0.0, 'y': 0.0},
                           reparameterisations={
                               'scale': {'parameters': ['x', 'y'],
                                         'scale': [5, 10]}},
                           proposal_plots='min',
                           analytic_priors=False, seed=1234)
