
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


class NFlow(BaseFlow):
    """
    Base class for flow objects implemented according to outline in nflows.
    These take an instance of a transform and a distribution from nflows.

    This replaces `Flow` from nflows. It removes the context and includes
    additional methods which are called in FlowModel.

    Parameters
    ----------
    transform : :obj: `nflows.transforms.Transform`
        Object that applys the transformation, must have`forward` and
        `inverse` methods. See nflows for more details.
    distribution : :obj: `nflows.distributions.Distribution`
        Object the serves as the base distribution used when sampling
        and computing the log probrability. Must have `log_prob` and
        `sample` methods. See nflows for details
    """
    def __init__(self, transform, distribution):
        super().__init__()

        for method in ['forward', 'inverse']:
            if not hasattr(transform, method):
                raise RuntimeError(
                    f'Transform does not have `{method}` method')

        for method in ['log_prob', 'sample']:
            if not hasattr(distribution, method):
                raise RuntimeError(
                    f'Distribution does not have `{method}` method')

        self._transform = transform
        self._distribution = distribution

    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the latent
        space and log |J|
        """
        return self._transform.forward(x)

    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log |J| (not log probability)
        """
        return self._transform.inverse(z)

    def sample(self, num_samples, context=None):
        """
        Produces N samples in the data space by drawing from the base
        distribution and the applying the inverse transform.

        Does NOT need to be specified by the user
        """
        noise = self._distribution.sample(num_samples)

        samples, _ = self._transform.inverse(noise)

        return samples

    def log_prob(self, inputs, context=None):
        """
        Computes the log probability of the inputs samples by apply the
        transform.

        Does NOT need to specified by the user
        """
        noise, logabsdet = self._transform(inputs)
        log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

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
        z, log_J = self.forward(x)
        log_prob = self.base_distribution_log_prob(z)
        return z, log_prob + log_J

    def sample_and_log_prob(self, N, context=None):
        """
        Generates samples from the flow, together with their log probabilities
        in the data space log p(x) = log p(z) + log|J|.

        For flows, this is more efficient that calling `sample` and `log_prob`
        separately.
        """
        z, log_prob = self._distribution.sample_and_log_prob(N)

        samples, logabsdet = self._transform.inverse(z)

        return samples, log_prob - logabsdet
