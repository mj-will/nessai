# -*- coding: utf-8 -*-
"""
Augmented version of FlowProposal.
"""

import logging

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy import stats
from scipy.special import logsumexp

from .. import config
from ..livepoint import numpy_array_to_live_points
from ..utils.structures import get_subset_arrays
from .flowproposal import FlowProposal

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
        self._base_inverse_rescale = self.inverse_rescale
        self.rescale = self._augmented_rescale
        self.inverse_rescale = self._augmented_inverse_rescale
        self.augment_parameters = [f"e_{i}" for i in range(self.augment_dims)]
        self.parameters += self.augment_parameters
        self.prime_parameters += self.augment_parameters
        logger.info(f"augmented x space parameters: {self.parameters}")
        logger.info(
            f"Augmented x prime space parameters: {self.prime_parameters}"
        )
        self.augment_dist = stats.multivariate_normal(
            np.zeros(self.augment_dims), np.eye(self.augment_dims)
        )

    def update_flow_config(self):
        """Update the flow configuration dictionary"""
        super().update_flow_config()
        m = np.ones(self.rescaled_dims)
        m[-self.augment_dims :] = -1
        self.flow_config["mask"] = m

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
            for an in self.augment_parameters:
                x_prime[an] = np.zeros(x_prime.size)
        elif generate_augment == "gaussian":
            for an in self.augment_parameters:
                x_prime[an] = self.rng.standard_normal(x_prime.size)
        else:
            raise RuntimeError("Unknown method for generating augment samples")

        return x_prime, log_J

    def _augmented_inverse_rescale(self, x_prime, return_unit_hypercube=False):
        """Inverse rescale with augment parameters.

        Parameters
        ----------
        x_prime : array
            Structured array in X prime space with augment parameters in the
            fields.
        return_unit_hypercube : bool, optional
            Return the unit hypercube values. Not currently supported
        """
        if return_unit_hypercube:
            raise NotImplementedError(
                "Inverse rescaling with augmented parameters is not supported"
            )
        x, log_J = self._base_inverse_rescale(
            x_prime, return_unit_hypercube=return_unit_hypercube
        )

        # Augment parameters are not rescaled
        for an in self.augment_parameters:
            x[an] = x_prime[an].copy()
        return x, log_J

    def augmented_prior(self, x):
        """
        Log Gaussian for augmented variables.

        If self.marginalise_augment is True, log_prior is 0.
        """
        if self.marginalise_augment:
            return np.zeros(len(x))
        else:
            x_aug = rfn.structured_to_unstructured(x[self.augment_parameters])
            return self.augment_dist.logpdf(x_aug)

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

        x_prime[:, -self.augment_dims :] = self.rng.standard_normal(
            (x_prime.shape[0], self.augment_dims)
        )

        _, log_prob = self.flow.forward_and_log_prob(x_prime)

        log_prob_e = np.sum(
            stats.norm.logpdf(x_prime[:, -self.augment_dims :]), axis=1
        )

        return -np.log(self.n_marg) + logsumexp(
            (log_prob - log_prob_e).reshape(-1, self.n_marg), axis=1
        )

    def backward_pass(
        self,
        z,
        rescale=True,
        discard_nans=True,
        return_unit_hypercube=False,
        return_z=False,
    ):
        """
        A backwards pass from the model (latent -> real).

        Parameters
        ----------
        z : array_like
            Structured array of points in the latent space.
        rescale : bool, optional (True)
            Apply inverse rescaling function.
        discard_nans : bool, optional (True)
            Discard samples with NaN log probability.
        return_unit_hypercube : bool, optional (False)
            Return samples in the unit hypercube.
        return_z : bool, optional (False)
            Return the latent samples.

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
        except AssertionError as e:
            logger.warning(
                "Assertion error raised when sampling from the flow."
                f"Error: {e}"
            )
            if return_z:
                return np.array([]), np.array([]), np.array([])
            else:
                return np.array([]), np.array([])

        if self.marginalise_augment:
            log_prob = self._marginalise_augment(x)

        if discard_nans:
            valid = np.isfinite(log_prob)
            x, log_prob, z = get_subset_arrays(valid, x, log_prob, z)

        x = numpy_array_to_live_points(
            x.astype(config.livepoints.default_float_dtype),
            self.prime_parameters,
        )
        # Apply rescaling in rescale=True
        if rescale:
            x, log_J = self.inverse_rescale(
                x, return_unit_hypercube=return_unit_hypercube
            )
            # Include Jacobian for the rescaling
            log_prob -= log_J
            if not return_unit_hypercube:
                x, log_prob, z = self.check_prior_bounds(x, log_prob, z)
        if return_z:
            return x, log_prob, z
        else:
            return x, log_prob
