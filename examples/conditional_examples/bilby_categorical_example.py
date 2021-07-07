#!/usr/bin/env python

import bilby
import numpy as np

outdir = './outdir/'
label = 'bilby_categorical_example'

bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level='INFO')


class TwoModelLikelihood(bilby.Likelihood):

    def __init__(self, n_samples=32, noise_scale=0.5):
        """
        A very simple Gaussian likelihood
        """
        super().__init__(parameters={'m': None, 'c': None, 'k': None})
        self.truth = {'m': 2.3, 'c': 0.5, 'model': 0}
        self.x = np.linspace(0, 1.5, n_samples)
        self.data = \
            self.truth['m'] * self.x + self.truth['c'] + \
            noise_scale * np.random.randn(n_samples)

    def log_likelihood(self):
        """Returns log likelihood of given live point.

        This is where we define the two models.
        """
        if self.parameters['k'] == 0:
            # Linear model
            line = self.parameters['m'] * self.x + self.parameters['c']
        elif self.parameters['k'] == 1:
            # Non-linear model
            line = self.parameters['m'] * (self.x ** 0.5) + \
                   self.parameters['c']
        else:
            raise RuntimeError(f"Invalid value for w: {self.parameters['k']}")

        log_l = -0.5 * np.sum((line - self.data) ** 2)
        return log_l


priors = dict(
    m=bilby.core.prior.Uniform(0, 5, 'm'),
    c=bilby.core.prior.Uniform(-1, 1, 'c'),
    k=bilby.core.prior.Categorical(2)
)

likelihood = TwoModelLikelihood()

flow_config = dict(
        max_epochs=50,
        patience=10,
        model_config=dict(n_blocks=2, n_neurons=4, n_layers=2,
                          device_tag='cpu',
                          kwargs=dict(batch_norm_between_layers=True))
        )

result = bilby.run_sampler(
    outdir=outdir,
    label=label,
    resume=False,
    plot=True,
    likelihood=likelihood,
    priors=priors,
    sampler='nessai',
    nlive=2000,
    maximum_uninformed=2000,
    flow_config=flow_config,
    injection_parameters={'m': 2.3, 'c': 0.5, 'k': 0},
    analytic_priors=True, seed=1234,
    flow_class='ConditionalFlowProposal',
    categorical_parameters=['k'],
    reparameterisations={
        'm': {'reparameterisation': 'default'},
        'c': {'reparameterisation': 'default'}
    },
    )
