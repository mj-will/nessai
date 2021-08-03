# -*- coding: utf-8 -*-
"""
Proposal method for initial sampling when priors are not analytical.
"""
import numpy as np

from .analytic import AnalyticProposal


class RejectionProposal(AnalyticProposal):
    """Object for rejection sampling from the priors.

    See parent for explanation of arguments and keyword arguments.

    Will be used when ``nessai`` is called with ``analytic_priors=False``. This
    is the default behaviour.

    Relies on :py:meth:`nessai.model.Model.new_point` to draw new points and
    :py:meth:`nessai.model.Model.new_point_log_prob` when computing the
    probability of each new point.
    """
    def __init__(self, *args, **kwargs):
        super(RejectionProposal, self).__init__(*args, **kwargs)
        self._checked_population = True
        self.population_acceptance = None

    def draw_proposal(self, N=None):
        """Draw new point(s).

        Parameters
        ----------
        N : int, optional
            Number of samples to draw. If not specified ``poolsize`` will be
            used.

        Returns
        -------
        structured_array
            Array of N new points
        """
        if N is None:
            N = self.poolsize
        return self.model.new_point(N=N)

    def log_proposal(self, x):
        """
        Log proposal probability. Calls \
                :meth:`nessai.model.Model.new_point_log_prob`

        Parameters
        ----------
        x : structured_array
            Array of new points

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of log-probabilites.
        """
        return self.model.new_point_log_prob(x)

    def compute_weights(self, x):
        """
        Get weights for the samples.

        Computes the log weights for rejection sampling sampling such that
        that the maximum log probability is zero.

        Parameters
        ----------
        x :  structured_array
            Array of points

        Returns
        -------
        log_w : :obj:`numpy.ndarray`
            Array of log-weights rescaled such that the maximum value is zero.
        """
        x['logP'] = self.model.log_prior(x)
        log_q = self.log_proposal(x)
        log_w = x['logP'] - log_q
        log_w -= np.nanmax(log_w)
        return log_w

    def populate(self, N=None):
        """
        Populate the pool by drawing from the proposal distribution and
        using rejection sampling.

        Will also evaluate the likelihoods if the proposal contains a
        multiprocessing pool.

        Parameters
        ----------
        N : int, optional
            Number of samples to draw. Not all samples will be accepted to
            the number of samples saved will be less than N. If not specified
            ``poolsize`` will be used.
        """
        if N is None:
            N = self.poolsize
        x = self.draw_proposal(N=N)
        log_w = self.compute_weights(x)
        log_u = np.log(np.random.rand(N))
        indices = np.where((log_w - log_u) >= 0)[0]
        self.samples = x[indices]
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        self.population_acceptance = self.samples.size / N
        if self.pool is not None:
            self.evaluate_likelihoods()
        self.populated = True
        self._checked_population = False
