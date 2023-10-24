# -*- coding: utf-8 -*-
"""
Base objects for implementing normalising flows.
"""
from abc import ABC, abstractmethod
from torch.nn import Module


class BaseFlow(Module, ABC):
    """
    Base class for all normalising flows.

    If implementing flows using distributions and transforms see NFlow.
    """

    device = None

    def to(self, device):
        """Wrapper that stores the device before moving the flow"""
        self.device = device
        super().to(device)

    @abstractmethod
    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the
        latent space and the log-Jacobian determinant.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the latent space
        :obj:`torch.Tensor`
            Tensor of log determinants of the Jacobian of the forward
            transformation
        """
        raise NotImplementedError()

    @abstractmethod
    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and the log-Jacobian determinant.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        :obj:`torch.Tensor`
            Tensor of log determinants of the Jacobian of the forward
            transformation
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, n, context=None):
        """
        Generate n samples in the data space

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        """
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x, context=None):
        """
        Compute the log probability for a set of samples in the data space

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_latent_distribution(self, n, context=None):
        """Sample from the latent distribution."""
        raise NotImplementedError

    @abstractmethod
    def base_distribution_log_prob(self, z, context=None):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of log probabilities of the latent samples
        """
        raise NotImplementedError()

    @abstractmethod
    def forward_and_log_prob(self, x, context=None):
        """
        Apply the forward transformation and compute the log probability
        of each sample

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the latent space
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_and_log_prob(self, n, context=None):
        """
        Generates samples from the flow, together with their log probabilities
        in the data space ``log p(x) = log p(z) + log|J|``.

        For flows, this is more efficient that calling ``sample`` and
        ``log_prob`` separately.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        raise NotImplementedError()

    def finalise(self):
        """Finalise the flow after training.

        Will be called after training the flow and loading the best weights.
        For example, can be used to finalise the Monte Carlo estimate of the
        normalising constant used in a LARS based flow.

        By default does nothing and should be implemented by the user.
        """
        pass

    def end_iteration(self):
        """Update the model at the end of an iteration.

        Will be called between training and validation.

        By default does nothing and should be overridden by an class that
        inherit from this class.
        """
        pass

    @abstractmethod
    def freeze_transform(self):
        """Freeze the transform part of the flow.

        Must be implemented by the child class.
        """
        raise NotImplementedError()

    @abstractmethod
    def unfreeze_transform(self):
        """Unfreeze the transform part of the flow.

        Must be implemented by the child class.
        """
        raise NotImplementedError()


class NFlow(BaseFlow):
    """Base class for flow objects from glasflow.nflows.

    This replaces `Flow` from glasflow.nflows. It includes additional methods
    which are called in FlowModel.

    Parameters
    ----------
    transform : :obj: `glasflow.nflows.transforms.Transform`
        Object that applies the transformation, must have`forward` and
        `inverse` methods. See glasflow.nflows for more details.
    distribution : :obj: `glasflow.nflows.distributions.Distribution`
        Object the serves as the base distribution used when sampling
        and computing the log probability. Must have `log_prob` and
        `sample` methods. See glasflow.nflows for details
    """

    def __init__(self, transform, distribution):
        super().__init__()
        from glasflow.nflows.transforms import Transform
        from glasflow.nflows.distributions import Distribution

        if not isinstance(transform, Transform):
            raise TypeError(
                "transform must inherit from "
                "`glasflow.nflows.transforms.Transform`. "
                f"Got: {type(transform)}"
            )

        if not isinstance(distribution, Distribution):
            raise TypeError(
                "distribution must inherit from "
                "`glasflow.nflows.transforms.Distribution`. "
                f"Got: {type(transform)}"
            )

        self._transform = transform
        self._distribution = distribution

    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the latent
        space and log-Jacobian determinant.
        """
        return self._transform.forward(x, context=context)

    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log-Jacobian determinant (not log probability).
        """
        return self._transform.inverse(z, context=context)

    def sample(self, num_samples, context=None):
        """
        Produces N samples in the data space by drawing from the base
        distribution and the applying the inverse transform.

        Does NOT need to be specified by the user
        """
        noise = self._distribution.sample(num_samples)

        samples, _ = self._transform.inverse(noise, context=context)

        return samples

    def log_prob(self, inputs, context=None):
        """
        Computes the log probability of the inputs samples by apply the
        transform.

        Does NOT need to specified by the user
        """
        noise, logabsdet = self._transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def sample_latent_distribution(self, n, context=None):
        if context is not None:
            raise NotImplementedError
        return self._distribution.sample(n)

    def base_distribution_log_prob(self, z, context=None):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow.
        """
        return self._distribution.log_prob(z)

    def forward_and_log_prob(self, x, context=None):
        """
        Apply the forward transformation and compute the log probability
        of each sample

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the latent space
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        z, log_J = self.forward(x, context=context)
        log_prob = self.base_distribution_log_prob(z)
        return z, log_prob + log_J

    def sample_and_log_prob(self, N, context=None):
        """
        Generates samples from the flow, together with their log probabilities
        in the data space ``log p(x) = log p(z) + log|J|``.

        For flows, this is more efficient that calling ``sample`` and
        ``log_prob`` separately.
        """
        z, log_prob = self._distribution.sample_and_log_prob(N)

        samples, logabsdet = self._transform.inverse(z, context=context)

        return samples, log_prob - logabsdet

    def finalise(self):
        """Finalise the flow after training.

        Checks if the base distribution or transform have finalise methods
        and calls them.
        """
        if hasattr(self._distribution, "finalise"):
            self._distribution.finalise()
        if hasattr(self._transform, "finalise"):
            self._transform.finalise()

    def end_iteration(self):
        """Update the model at the end of an iteration.

        Will be called between training and validation.
        """
        if hasattr(self._distribution, "end_iteration"):
            self._distribution.end_iteration()
        if hasattr(self._transform, "end_iteration"):
            self._transform.end_iteration()

    def freeze_transform(self):
        """Freeze the transform part of the flow"""
        self._transform.requires_grad_(False)

    def unfreeze_transform(self):
        """Unfreeze the transform part of the flow"""
        self._transform.requires_grad_(True)
