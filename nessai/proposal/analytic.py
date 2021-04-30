# -*- coding: utf-8 -*-
"""
Proposal method for initial sampling when priors can be sampled analytically.
"""
import datetime

import numpy as np

from .base import Proposal


class AnalyticProposal(Proposal):
    """
    Class for drawing samples from analytic priors.

    This assumes the `new_point` method of the model draws points
    from the prior
    """
    def __init__(self, *args, poolsize=1000, **kwargs):
        super(AnalyticProposal, self).__init__(*args, **kwargs)
        self.populated = False
        self._poolsize = poolsize

    @property
    def poolsize(self):
        """Poolsize used for drawing new samples in batches."""
        return self._poolsize

    def populate(self, N=None):
        """
        Populate the pool by drawing from the priors
        """
        if N is None:
            N = self.poolsize
        self.samples = self.model.new_point(N=N)
        self.samples['logW'] = np.zeros(N)
        self.samples['logP'] = self.model.log_prior(self.samples)
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        if self.pool is not None:
            self.evaluate_likelihoods()
        self.populated = True

    def draw(self, old_sample, **kwargs):
        """
        Propose a new sample. Draws from the pool if it is populated, else
        it populates the pool.

        Parameters
        ----------
        old_sample : structured_array
            Old sample, this is not used in the proposal method
        kwargs :
            Keyword arguments passed to ``populate``.
        """
        if not self.populated:
            st = datetime.datetime.now()
            self.populate(**kwargs)
            self.population_time += (datetime.datetime.now() - st)
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
        return new_sample
