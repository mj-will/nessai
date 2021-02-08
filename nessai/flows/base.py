
from torch.nn import Module


class BaseFlow(Module):
    """
    Base class for all normalising flows.

    If implementing flows using distributions and transforms see NFlow.
    """
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def sample(self, n, context=None):
        """
        Generate n samples in the data space

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the data space
        """
        raise NotImplementedError()

    def log_prob(self, x, context=None):
        """
        Compute the log probability for a set of samples in the data space

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
