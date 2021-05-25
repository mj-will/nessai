#!/usr/bin/env python
# Example of using nessai with the eggbox function. Nessai is not well
# suited to this sorted problem. The augmented proposal method is used to
# to improve the results but ultimately likelihoods that this multi-modal
# are a challenge.

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger


output = './outdir/eggbox/'
logger = setup_logger(output=output, log_level='INFO')


class EggboxModel(Model):
    """
    Eggbox problem from https://arxiv.org/pdf/0809.3437v1.pdf.

    Based on the example in cpnest: \
        https://github.com/johnveitch/cpnest/blob/master/examples/eggbox.py
    """

    def __init__(self, dims):
        self.names = [str(i) for i in range(dims)]
        self.bounds = {n: [0, 10 * np.pi] for n in self.names}

    def log_prior(self, x):
        log_p = 0.
        for n in self.names:
            log_p += (np.log((x[n] >= self.bounds[n][0])
                             & (x[n] <= self.bounds[n][1]))
                      - np.log(self.bounds[n][1] - self.bounds[n][0]))
        return log_p

    def log_likelihood(self, x):
        log_l = 1.0
        for n in self.names:
            log_l *= np.cos(x[n] / 2.)
        return (log_l + 2.0) ** 5.0


flow_config = dict(
        batch_size=1000,
        max_epochs=200,
        patience=50,
        model_config=dict(n_blocks=4, n_neurons=8, n_layers=2, ftype='realnvp',
                          kwargs=dict(batch_norm_between_layers=False,
                                      linear_transform='lu'))
        )


fs = FlowSampler(EggboxModel(2), output=output, flow_config=flow_config,
                 resume=False, seed=1234, flow_class='AugmentedFlowProposal',
                 augment_features=4, expansion_fraction=2.0, nlive=2000,
                 poolsize=2000, maximum_uninformed=np.inf,
                 update_poolsize=True, max_threads=2, n_pool=1)

fs.run()
