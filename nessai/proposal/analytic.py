import datetime

import numpy as np

from .base import Proposal


class AnalyticProposal(Proposal):
    """"
    Class for drawining from analytic priors

    This assumes the `new_point` method of the model draws points
    from the prior
    """
    def __init__(self, *args, **kwargs):
        super(AnalyticProposal, self).__init__(*args, **kwargs)
        self.populated = False

    def populate(self, N=1000):
        """
        Populate the pool by drawing from the priors
        """
        self.samples = self.model.new_point(N=N)
        self.samples['logP'] = self.model.log_prior(self.samples)
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        if self.pool is not None:
            self.evaluate_likelihoods()
        self.populated = True

    def draw(self, old_sample):
        """
        Propose a new sample. Draws from the pool if it is populated, else
        it populates the pool.

        Parameters
        ----------
        old_sample : structured_array
            Old sample, this is not used in the proposal method
        """
        if not self.populated:
            st = datetime.datetime.now()
            self.populate()
            self.population_time += (datetime.datetime.now() - st)
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
        return new_sample
