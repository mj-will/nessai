# -*- coding: utf-8 -*-
"""
Proposal method for Gaussian likelihoods. This proposal is designed for
testing nessai.
"""

import numpy as np
from scipy.stats import chi

from ..proposal.base import Proposal
from ..livepoint import (
    live_points_to_array,
    numpy_array_to_live_points
)
from ..utils import (
    draw_nsphere,
    draw_truncated_gaussian,
    volume_nball,
    area_sphere,
)


class ExactProposal(Proposal):
    """Base class for proposals used for testing nessai.

    These proposals are designed to be used with a Gaussian likelihood.
    """
    def __init__(self, model, sampling_method='rejection_sampling', **kwargs):
        super().__init__(model, **kwargs)
        if sampling_method == 'rejection_sampling':
            self._rejection_sampling = True
        else:
            self._rejection_sampling = False

    def draw(self, old_sample, **kwargs):
        """Proposal a new sample"""
        if self.draw_priors:
            x = self.model.new_point()
            x['logP'] = self.model.log_prior(x)
            if not self._rejection_sampling:
                x['logW'] = x['logP'] - self.model.new_point_log_prob(x)
            else:
                x['logW'] = 0.0
            return x
        else:
            sample = live_points_to_array(old_sample, self.model.names)
            r = np.sqrt(np.sum(sample ** 2))
            x = self._sample(r)
            x_live = numpy_array_to_live_points(x, self.model.names)
            x_live['logP'] = self.model.log_prior(x_live)
            if not self._rejection_sampling:
                x_live['logW'] = x_live['logP'] - \
                                 self.log_proposal_prob(x, r_max=r)
            else:
                x_live['logW'] = 0.0
            return x_live[0]


class ExactUniformProposal(ExactProposal):
    """
    Proposal the samples uniformly within contours of equal likelihood for
    an n-dimensional Gaussian.
    """
    def log_proposal_prob(self, x, r_max=None):
        """The log proposal probability"""
        log_p = -np.log(volume_nball(self.model.dims, r=r_max))
        return log_p

    def _sample(self, r, N=1):
        return draw_nsphere(self.model.dims, r=r, N=N)


class ExactGaussianProposal(ExactProposal):
    """
    Proposal that draws normally distributed samples within contours of equal
    likelihood for an n-dimensional Gaussian.
    """
    def initialise(self):
        """Initialise the proposal"""
        super().initialise()
        self._chi = chi(self.model.dims)

    def log_proposal_prob(self, x, r_max=None):
        """The log proposal probability"""
        r = np.sqrt(np.sum(x ** 2, axis=-1))
        log_p = self._chi.logpdf(r) - self._chi.logcdf(r_max)
        log_norm = np.log(area_sphere(self.model.dims, r=r))
        return log_p - log_norm

    def _sample(self, r, N=1):
        return draw_truncated_gaussian(self.model.dims, r=r, N=N)
