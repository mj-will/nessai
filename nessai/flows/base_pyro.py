
import torch
from torch.distributions.utils import _sum_rightmost

from .base import BaseFlow


class PyroFlow(BaseFlow):
    """
    Base class for normalising flows implemented in Pyro

    Parameters
    ----------
    flow_dist : :obj:`TransformDistribution`
        Normalising flow implemented in Pyro using the
        `TransformedDistribution` object.
    """
    def __init__(self, flow_dist):
        super().__init__()
        self._flow_dist = flow_dist

    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the
        latent space and log |J|

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the latent space
        :obj:`torch.Tensor`
            Tensor of log determinants of the Jacobian of the forward
            transformation
        """
        if context is not None:
            raise ValueError
        event_dim = len(self._flow_dist.event_shape)
        log_jacobian = 0.0
        z = x
        for transform in reversed(self._flow_dist.transforms):
            x = transform.inv(z)
            log_jacobian = log_jacobian - \
                _sum_rightmost(transform.log_abs_det_jacobian(x, z),
                               event_dim - transform.event_dim)
            z = x

        return z, log_jacobian

    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log |J|

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        :obj:`torch.Tensor`
            Tensor of log determinants of the Jacobian of the forward
            transformation
        """
        if context is not None:
            raise ValueError
        event_dim = len(self._flow_dist.event_shape)
        log_jacobian = 0.0
        for transform in reversed(self._flow_dist.transforms):
            x = transform(z)
            log_jacobian = log_jacobian + \
                _sum_rightmost(transform.log_abs_det_jacobian(x, z),
                               event_dim - transform.event_dim)
            z = x

        return x, log_jacobian

    def sample(self, n, context=None):
        """
        Generate n samples in the data space

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        """
        if context is not None:
            raise ValueError
        return self._flow_dist.sample((n,))

    def log_prob(self, x, context=None):
        """
        Compute the log probability for a set of samples in the data space

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        if context is not None:
            raise ValueError
        return self._flow_dist.log_prob(x)

    def base_distribution_log_prob(self, z, context=None):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of log probabilities of the latent samples
        """
        if context is not None:
            raise ValueError
        self._flow_dist.base_dist.log_prob(z)

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
        if context is not None:
            raise ValueError
        event_dim = len(self._flow_dist.event_shape)
        log_prob = 0.0
        z = x
        for transform in reversed(self._flow_dist.transforms):
            x = transform.inv(z)
            log_prob = log_prob - \
                _sum_rightmost(transform.log_abs_det_jacobian(x, z),
                               event_dim - transform.event_dim)
            z = x

        log_prob = log_prob + \
            _sum_rightmost(self._flow_dist.base_dist.log_prob(z),
                           event_dim - len(self.base_dist.event_shape))
        return z, log_prob

    def sample_and_log_prob(self, n, context=None):
        """
        Generates samples from the flow, together with their log probabilities
        in the data space log p(x) = log p(z) + log|J|.

        For flows, this is more efficient that calling `sample` and `log_prob`
        separately.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        if context is not None:
            raise ValueError
        event_dim = len(self._flow_dist.event_shape)
        with torch.no_grad():
            z = self._flow_dist.base_dist.sample((n,))
            log_prob = _sum_rightmost(
                self._flow_dist.base_dist.log_prob(z),
                event_dim - len(self.base_dist.event_shape))

            for transform in reversed(self._flow_dist.transforms):
                x = transform(z)
                log_prob = log_prob + \
                    _sum_rightmost(transform.log_abs_det_jacobian(x, z),
                                   event_dim - transform.event_dim)
                z = x

        return x, log_prob
