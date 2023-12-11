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
            Array of log-probabilities.
        """
        return self.model.new_point_log_prob(x)

    def compute_weights(self, x, return_log_prior=False):
        """
        Get weights for the samples.

        Computes the log weights for rejection sampling sampling but does not
        normalize the weights.

        Parameters
        ----------
        x :  structured_array
            Array of points
        return_log_prior: bool
            If true, the log-prior probability is also returned.

        Returns
        -------
        log_w : :obj:`numpy.ndarray`
            Array of log-weights rescaled such that the maximum value is zero.
        """
        log_p = self.model.batch_evaluate_log_prior(x)
        log_q = self.log_proposal(x)
        log_w = log_p - log_q
        if return_log_prior:
            return log_w, log_p
        else:
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
        log_w, x["logP"] = self.compute_weights(x, return_log_prior=True)
        log_w -= np.nanmax(log_w)
        log_u = np.log(np.random.rand(N))
        indices = np.where((log_w - log_u) >= 0)[0]
        self.samples = x[indices]
        self.indices = np.random.permutation(self.samples.shape[0]).tolist()
        self.population_acceptance = self.samples.size / N
        self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
            self.samples
        )
        self.populated = True
        self._checked_population = False
