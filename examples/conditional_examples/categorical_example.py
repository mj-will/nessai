#!/usr/bin/env python

# Example of using categorical parameters with nessai.
# Credit to Greg Ashton for proposing this idea with BilbyMCMC

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal

from nessai.evidence import recompute_evidence
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_array, numpy_array_to_live_points
from nessai.model import Model
from nessai.utils import setup_logger


output = './outdir/categorical_example/'
logger = setup_logger(output=output, log_level='INFO')


class CategoricalModel(Model):
    """Model with categorical parameters.

    Data is modelled as line with Gaussian noise

    There are two possible versions of the likelihood:
    0.
    1. Non-linear model mx^0.5 + c
    A weight parameter `w` allows the sampler to move between likelihoods
    and this is set as a categorical parameter in the model.
    """
    def __init__(self, dims=2, models=[0, 1]):
        names = [f'x_{i}' for i in range(dims)]
        self.names = names + ['model']
        self.bounds = {n:  [-10, 10] for n in self.names}
        self.bounds['model'] = [0, 1]
        self.truth = {'model': 0}
        self.categorical = {'model': (0, 1)}
        self.true_bayes_factor = 2.0
        self.allowed_models = models
        logger.warning(f'True Bayes factor: {self.true_bayes_factor}')
        self._model = multivariate_normal(mean=dims * [0], cov=np.eye(dims))

        gaussian_lnZ = np.sum([-np.log(np.ptp(self.bounds[n])) for n in names])
        self.lnZ_model_0 = np.log(2 / 3) + gaussian_lnZ
        self.lnZ_model_1 = np.log(1 / 3) + gaussian_lnZ

    def new_point(self, N=1):
        """Draw new samples.

        Since we have a categorical parameter we need to refine this method,
        the default version trys to sample uniformly between the prior bounds.
        """
        x = numpy_array_to_live_points(np.zeros([N, self.dims]), self.names)
        for n in self.names[:-1]:
            x[n] = np.random.uniform(*self.bounds[n], N)
        x['model'] = np.random.choice(self.allowed_models, size=N)
        return x

    def new_point_log_prob(self, x):
        """Probability for the `new_point` method."""
        return self.log_prior(x)

    def log_prior(self, x):
        """Returns log of prior given a live point.

        We assume a uniform prior on m, c and the model parameter.
        """
        return np.log(len(self.allowed_models)) * np.ones(x.size)

    def log_likelihood(self, x):
        """Returns log likelihood of given live point.

        This is where we define the two models.
        """
        _x = live_points_to_array(x, self.names[:-1])
        log_l = self._model.logpdf(_x)
        if x['model'] == 0:
            return np.log(2 / 3) + log_l
        elif x['model'] == 1:
            return np.log(1 / 3) + log_l
        else:
            raise RuntimeError(f"Invalid value for w: {x['model']}")


flow_config = dict(
        batch_size=1000,
        max_epochs=200,
        patience=20,
        model_config=dict(n_blocks=2, n_neurons=4, n_layers=2,
                          kwargs=dict(linear_transform=None))
        )

# Can change this to run inferences on a single model.
model = CategoricalModel(dims=2, models=[0, 1])

# Configure the sampler
# Make sure we're using 'ConditionalFlowProposal'
fp = FlowSampler(
    model,
    output=output,
    flow_config=flow_config,
    resume=False,
    seed=1234,
    flow_class='ConditionalFlowProposal',
    nlive=4000,
    maximum_uninformed=8000,
    poolsize=4000,
    reparameterisations={
        'default': {'parameters': model.names[:-1]}
    }
)

# And go!
fp.run()

logger.info(f'True ln Z for model 0: {model.lnZ_model_0}')
logger.info(f'True ln Z for model 0: {model.lnZ_model_1}')

# Compute Bayes Factors
nested_samples = np.asarray(fp.nested_samples)

m0_its = (nested_samples['model'] == 0)
m1_its = (nested_samples['model'] == 1)

ns_m0 = nested_samples[m0_its]
ns_m1 = nested_samples[m1_its]

nlive = np.array(fp.ns.categorical_history['model'])
nlive_0 = nlive[:, 0][m0_its]
nlive_1 = nlive[:, 1][m1_its]

lnZ_0, dlnZ_0 = recompute_evidence(ns_m0['logL'], nlive_0)
lnZ_1, dlnZ_1 = recompute_evidence(ns_m1['logL'], nlive_1)

lnBF = lnZ_0 - lnZ_1
dlnBF = np.sqrt(dlnZ_0 ** 2 + dlnZ_1 ** 2)
BF = np.exp(lnBF)
dBF = BF * dlnBF

logger.info(f'True Bayes factor: {model.true_bayes_factor}')
logger.info(f'True ln Bayes Factor: {np.log(model.true_bayes_factor):.3f}')

logger.info(f'Bayes factor using evidences: {BF:.3f} +/- {dBF:.3f}')
logger.info(f'ln Bayes factor using evidences: {lnBF:.3f} +/- {dlnBF:.3f}')

# Posterior samples
posterior_samples = fp.posterior_samples
# Ratio of samples in the posterior
w = posterior_samples['model']
m0 = (w == 0).sum()
m1 = (w == 1).sum()
ratio = m0 / m1
ratio_err = ratio * np.sqrt((m0 + m1) / (m0 * m1))
logger.info('Number of samples in posterior:')
logger.info(f'Model 0: {m0}')
logger.info(f'Model 1: {m1}')
logger.info(f'Ratio using samples: {ratio:.3f} +/- {ratio_err:.3f}')

# Plot posteriors using Seaborn and hue argument.
df = pd.DataFrame(posterior_samples, columns=model.names)
fig = sns.pairplot(
    data=df,
    hue='model',
    kind='kde',
    corner=True
)
fig.savefig(f'{output}/posterior_comparison.png')
