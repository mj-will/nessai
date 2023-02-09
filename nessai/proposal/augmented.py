# -*- coding: utf-8 -*-
"""
Augmented version of FlowProposal.
"""
import logging

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from .. import config
from .flowproposal import FlowProposal
from ..livepoint import numpy_array_to_live_points

logger = logging.getLogger(__name__)


class AugmentedFlowProposal(FlowProposal):
    """Version of FlowProposal that uses AugmentedFlows.

    Augmented normalising flows were proposed in: \
        https://arxiv.org/abs/2002.07101 and add auxiliary parameters to the
    inputs of the flow which are drawn from a Gaussian. This improves the
    flows' ability to learn multimodal distribution.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User defined model
    augment_dims : int
        Number of augment parameters to add to the inputs
    generate_augment : {'gaussian', 'zeroes', 'zeros'}, optional
        Method used when computing the radius of the latent contour.
    marginalise_augment : bool, optional
        Use the marginalised likelihood when performing rejection sampling.
        Adds significant computation cost.
    n_marg : int, optional
        Number of samples to use when approximating the marginalised
        likelihood.
    """

    def __init__(
        self,
        model,
        augment_dims=1,
        generate_augment="gaussian",
        marginalise_augment=False,
        n_marg=50,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.augment_dims = augment_dims
        self.generate_augment = generate_augment
        self.marginalise_augment = marginalise_augment
        self.n_marg = n_marg

    def set_rescaling(self):
        """Configure the rescaling.

        Calls the method from the parent class first and then adds the
        auxiliary parameters.
        """
        super().set_rescaling()
        # Cannot use super().rescale because rescale is changed in
        # set rescaling.
        self._base_rescale = self.rescale
        self.rescale = self._augmented_rescale
        self.augment_names = [f"e_{i}" for i in range(self.augment_dims)]
        self.names += self.augment_names
        self.rescaled_names += self.augment_names
        logger.info(f"augmented x space parameters: {self.names}")
        logger.info(f"parameters to rescale {self.rescale_parameters}")
        logger.info(
            f"Augmented x prime space parameters: {self.rescaled_names}"
        )

    def update_flow_config(self):
        """Update the flow configuration dictionary"""
        super().update_flow_config()
        m = np.ones(self.rescaled_dims)
        m[-self.augment_dims :] = -1
        if "kwargs" not in self.flow_config["model_config"].keys():
            self.flow_config["model_config"]["kwargs"] = {}
        self.flow_config["model_config"]["kwargs"]["mask"] = m

    def _augmented_rescale(
        self, x, generate_augment=None, compute_radius=False, **kwargs
    ):
        """Rescale with augment parameter.

        Parameters
        ----------
        x : array
            Structured array in X space with augment parameters in the
            fields.
        generate_augment : {None, 'gaussian', 'zeros', 'zeroes', False}, \
                optional
            Method used to generate the augmented parameters. If None Gaussian
            is used if compute_radius=False else attribute of the same name
            is used. If False no values are generated.
        compute_radius : bool, optional
            Boolean to indicate when rescale is being used for computing a
            radius.
        """
        x_prime, log_J = self._base_rescale(
            x, compute_radius=compute_radius, **kwargs
        )

        if generate_augment is None:
            if compute_radius:
                generate_augment = self.generate_augment
            else:
                generate_augment = "gaussian"

        if generate_augment in ["zeros", "zeroes"]:
            for an in self.augment_names:
                x_prime[an] = np.zeros(x_prime.size)
        elif generate_augment == "gaussian":
            for an in self.augment_names:
                x_prime[an] = np.random.randn(x_prime.size)
        else:
            raise RuntimeError("Unknown method for generating augment samples")

        return x_prime, log_J

    def augmented_prior(self, x):
        """
        Log Gaussian for augmented variables.

        If self.marginalise_augment is True, log_prior is 0.
        """
        log_p = 0.0
        if not self.marginalise_augment:
            for n in self.augment_names:
                log_p += stats.norm.logpdf(x[n])
        return log_p

    def log_prior(self, x):
        """
        Compute the prior probability in the non-prime space.
        """
        return super().log_prior(x) + self.augmented_prior(x)

    def x_prime_log_prior(self, x):
        """
        Compute prior probability in the prime space.
        """
        return super().x_prime_log_prior(x) + self.augmented_prior(x)

    def _marginalise_augment(self, x_prime):
        """Marginalise out the augmented features.

        Note that x_prime is not a structured array. See the original paper
        for the details
        """
        x_prime = np.repeat(x_prime, self.n_marg, axis=0)

        x_prime[:, -self.augment_dims :] = np.random.randn(
            x_prime.shape[0], self.augment_dims
        )

        _, log_prob = self.flow.forward_and_log_prob(x_prime)

        log_prob_e = np.sum(
            stats.norm.logpdf(x_prime[:, -self.augment_dims :]), axis=1
        )

        return -np.log(self.n_marg) + logsumexp(
            (log_prob - log_prob_e).reshape(-1, self.n_marg), axis=1
        )

    def backward_pass(self, z, rescale=True):
        """
        A backwards pass from the model (latent -> real)

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space
        rescale : bool, optional (True)
            Apply inverse rescaling function

        Returns
        -------
        x : array_like
            Samples in the latent space
        log_prob : array_like
            Log probabilities corresponding to each sample (including the
            Jacobian)
        """
        try:
            x, log_prob = self.flow.sample_and_log_prob(
                z=z, alt_dist=self.alt_dist
            )
        except AssertionError:
            return np.array([]), np.array([])

        if self.marginalise_augment:
            log_prob = self._marginalise_augment(x)

        valid = np.isfinite(log_prob)
        x, log_prob = x[valid], log_prob[valid]
        x = numpy_array_to_live_points(
            x.astype(config.livepoints.default_float_dtype),
            self.rescaled_names,
        )
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(x)
            # Include Jacobian for the rescaling
            log_prob -= log_J
            x, log_prob = self.check_prior_bounds(x, log_prob)
        return x, log_prob
