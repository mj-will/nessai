# -*- coding: utf-8 -*-
"""
Proposal method for initial sampling when priors can be sampled analytically.
"""
import datetime

import numpy as np

from .base import Proposal


class AnalyticProposal(Proposal):
    """Class for drawing samples from analytic priors.

    Will be used when ``nessai`` is called with ``analytic_priors=True`` and
    assumes the :py:meth:`nessai.model.Model.new_point` samples directly
    from the prior. This method must be implemented by the user.

    Parameters
    ----------
    args :
        Arguments passed to the parent class.
    poolsize : int, optional
        Number of points drawn at once.
    kwargs :
        Keyword arguments passed to the parent class.
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
        Populate the pool by drawing from the priors.

        Will also evaluate the likelihoods if the proposal contains a
        multiprocessing pool.

        Parameters
        ----------
        N : int, optional
            Number of samples to draw. If not specified ``poolsize`` will be
            used.
        """
        if N is None:
            N = self.poolsize
        self.samples = self.model.new_point(N=N)
        self.samples["logP"] = self.model.batch_evaluate_log_prior(
            self.samples
        )
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
            self.samples
        )
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
            Keyword arguments passed to \
                :py:meth:`~nessai.proposal.analytic.AnalyticProposal.populate`
        """
        if not self.populated:
            st = datetime.datetime.now()
            self.populate(**kwargs)
            self.population_time += datetime.datetime.now() - st
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
        return new_sample
