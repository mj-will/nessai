#!/usr/bin/env python

# Example of using INS-nessai with Bilby (Requires separate installation)

import bilby
import numpy as np

outdir = './outdir/'
label = 'bilby_importance_example'

bilby.core.utils.setup_logger(outdir=outdir, label=label)


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


# Define priors (this provides the bounds that are then used in Nessai)
priors = dict(
    x=bilby.core.prior.Uniform(-10, 10, 'x'),
    y=bilby.core.prior.Uniform(-10, 10, 'y')
)

# Instantiate the likleihood
likelihood = SimpleGaussianLikelihood()

# Run using bilby.run_sampler, any kwargs are parsed to the sampler
# We set the sampler to `nessai_importance`.
result = bilby.run_sampler(
    outdir=outdir,
    label=label,
    resume=False,
    seed=1234,
    plot=True,
    injection_parameters={'x': 0.0, 'y': 0.0},
    likelihood=likelihood,
    priors=priors,
    sampler='nessai_importance',
    nlive=3000,
    min_samples=1000,
)
