import datetime

import numpy as np

from .base import Proposal


class RejectionProposal(Proposal):
    """
    Object for rejection sampling from the priors. Relies on the method
    `new_point` included in `flowproposal.model.Model`.

    Parameters
    ----------
    model : obj
        User-defined model
    poolsize : int
        Number of new samples to store in the pool
    """
    def __init__(self, model, poolsize=1000, **kwargs):
        super(RejectionProposal, self).__init__(model, **kwargs)
        self._poolsize = poolsize
        self.populated = False
        self._acceptance_checked = True

    @property
    def poolsize(self):
        return self._poolsize

    def draw_proposal(self):
        """Draw a signal new point"""
        return self.model.new_point(N=self.poolsize)

    def log_proposal(self, x):
        """
        Log proposal probability. Calls `model.new_point_log_prob`

        Parameters
        ----------
        x : structured_array
            Array of new points
        """
        return self.model.new_point_log_prob(x)

    def compute_weights(self, x):
        """
        Get weights for the samples.

        Computes the log weights for rejection sampling sampling such that
        that the maximum log probability is zero.

        Parameters
        ----------
        x :  structed_arrays
            Array of points
        """
        x['logP'] = self.model.log_prior(x)
        log_q = self.log_proposal(x)
        log_w = x['logP'] - log_q
        log_w -= np.nanmax(log_w)
        return log_w

    @property
    def population_acceptance(self):
        """
        Check the acceptance of the current proposal method.

        If this method has already been called since the proposal was
        last populated it returns None.
        """
        if self._acceptance_checked:
            return None
        else:
            self._acceptance_checked = True
            return self._population_acceptance

    @population_acceptance.setter
    def population_acceptance(self, acceptance):
        """
        Set the population acceptance and reset the flag
        """
        self._population_acceptance = acceptance
        self._acceptance_checked = False

    def populate(self, N=None):
        """
        Populate the pool by drawing from the proposal distribution and
        using rejection sampling.
        """
        if N is None:
            N = self.poolsize
        x = self.draw_proposal()
        log_w = self.compute_weights(x)
        log_u = np.log(np.random.rand(N))
        indices = np.where((log_w - log_u) >= 0)[0]
        self.samples = x[indices]
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        self.population_acceptance = self.samples.size / self.poolsize
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
