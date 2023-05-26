# -*- coding: utf-8 -*-
"""
Utilities to make interfacing with bilby easier.
"""
from inspect import signature
from typing import List, Callable

import bilby
import numpy as np

from ..livepoint import dict_to_live_points
from ..model import Model


class BilbyModel(Model):
    """Model class to wrap a bilby likelihood and prior dictionary.

    Parameters
    ----------
    priors : :obj:`bilby.core.prior.PriorDict`
        Bilby PriorDict object
    likelihood : :obj:`bilby.core.likelihood.Likelihood`
        Bilby Likelihood object
    use_ratio : bool
        Whether to use the log-likelihood ratio (True) or log-likelihood
        (False).
    """

    def __init__(
        self,
        *,
        priors: bilby.core.prior.PriorDict,
        likelihood: bilby.core.likelihood.Likelihood,
        use_ratio: bool = False,
    ):
        if not isinstance(priors, bilby.core.prior.PriorDict):
            raise TypeError("priors must be an instance of PriorDict")

        self.bilby_priors = priors
        self.bilby_likelihood = likelihood
        self.use_ratio = use_ratio
        self.names = self.bilby_priors.non_fixed_keys
        self._update_bounds()
        self.validate_bilby_likelihood()

    def _update_bounds(self):
        self.bounds = {
            key: [
                self.bilby_priors[key].minimum,
                self.bilby_priors[key].maximum,
            ]
            for key in self.names
        }

    def validate_bilby_likelihood(self) -> None:
        """Validate the bilby likelihood object"""
        theta = self.bilby_priors.sample()
        self.bilby_likelihood.parameters.update(theta)
        if self.use_ratio:
            self.bilby_likelihood.log_likelihood_ratio()
        else:
            self.bilby_likelihood.log_likelihood()

    def log_likelihood(self, x):
        """Compute the log likelihood"""
        theta = {n: x[n].item() for n in self.names}
        self.bilby_likelihood.parameters.update(theta)
        if self.use_ratio:
            return self.bilby_likelihood.log_likelihood_ratio()
        else:
            return self.bilby_likelihood.log_likelihood()

    def log_prior(self, x):
        """Compute the log prior.

        Also evaluates the likelihood constraints.
        """
        theta = {n: x[n] for n in self.names}
        return self.bilby_priors.ln_prob(theta, axis=0) + np.log(
            self.bilby_priors.evaluate_constraints(theta)
        )

    def new_point(self, N=1):
        """Draw a point from the prior"""
        prior_samples = self.bilby_priors.sample(size=N)
        samples = {n: prior_samples[n] for n in self.names}
        return dict_to_live_points(samples)

    def new_point_log_prob(self, x):
        """Proposal probability for new the point"""
        return self.log_prior(x)

    def from_unit_hypercube(self, x):
        """Map samples from the unit hypercube to the prior."""
        theta = {}
        for n in self.names:
            theta[n] = self.bilby_priors[n].rescale(x[n])
        return dict_to_live_points(theta)

    def to_unit_hypercube(self, x):
        """Map samples from the prior to the unit hypercube."""
        theta = {n: x[n] for n in self.names}
        return dict_to_live_points(self.bilby_priors.cdf(theta))


class BilbyModelLikelihoodConstraint(BilbyModel):
    """Bilby model where prior constraints are included in the likelihood."""

    def log_likelihood(self, x):
        """Compute the log likelihood.

        Also evaluates the likelihood constraints.
        """
        theta = {n: x[n].item() for n in self.names}
        if not self.bilby_priors.evaluate_constraints(theta):
            return -np.inf
        self.bilby_likelihood.parameters.update(theta)
        if self.use_ratio:
            return self.bilby_likelihood.log_likelihood_ratio()
        else:
            return self.bilby_likelihood.log_likelihood()

    def log_prior(self, x):
        """Compute the log prior."""
        theta = {n: x[n] for n in self.names}
        return self.bilby_priors.ln_prob(theta, axis=0)


def _get_standard_methods() -> List[Callable]:
    """Get a list of the methods used by the standard sampler"""
    from ..flowsampler import FlowSampler
    from ..proposal import AugmentedFlowProposal, FlowProposal
    from ..samplers import NestedSampler

    methods = [
        FlowSampler.run,
        AugmentedFlowProposal,
        FlowProposal,
        NestedSampler,
        FlowSampler,
    ]
    return methods


def get_all_kwargs() -> dict:
    """Get a dictionary of all possible kwargs and their default values.

    Returns
    -------
    Dictionary of kwargs and their default values.
    """
    methods = _get_standard_methods()
    kwargs = {}
    for m in methods:
        kwargs.update(
            {
                k: v.default
                for k, v in signature(m).parameters.items()
                if v.default is not v.empty
            }
        )
    return kwargs


def get_run_kwargs_list() -> List[str]:
    """Get a list of kwargs used in the run method"""
    from ..flowsampler import FlowSampler

    method = FlowSampler.run

    run_kwargs_list = [
        k
        for k, v in signature(method).parameters.items()
        if v.default is not v.empty
    ]
    return run_kwargs_list
