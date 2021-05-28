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
from ..utils import draw_nsphere, volume_nball


class AnalyticGaussianProposal(Proposal):
    """
    Proposal the samples uniformly within contours of equal likelihood for
    an n-dimensional Gaussian.
    """
    def __init__(self, model, sampling_method='rejection_sampling', **kwargs):
        super().__init__(model, **kwargs)
        if sampling_method == 'rejection_sampling':
            self._rejection_sampling = True
        else:
            self._rejection_sampling = False

    def initialise(self):
        """Initialise the proposal"""
        super().initialise()
        self._chi = chi(self.model.dims)

    def log_proposal_prob(self, x, r_max=None):
        """The log proposal probability"""
        log_p = -np.log(volume_nball(self.model.dims, r=r_max))
        return log_p

    def draw(self, old_sample, **kwargs):
        """Proposal a new sample"""
        if self.draw_priors:
            x = self.model.new_point()
            x['logP'] = self.model.log_prior(x)
            x['logW'] = 0.0
            return x
        else:
            sample = live_points_to_array(old_sample, self.model.names)
            r = np.sqrt(np.sum(sample ** 2))
            x = draw_nsphere(self.model.dims, r=r, N=1)
            x_live = numpy_array_to_live_points(x, self.model.names)
            x_live['logP'] = self.model.log_prior(x_live)
            if not self._rejection_sampling:
                x_live['logW'] = x_live['logP'] - \
                                 self.log_proposal_prob(x, r_max=r)
            else:
                x_live['logW'] = 0.0
            return x_live[0]
