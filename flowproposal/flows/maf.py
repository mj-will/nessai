
from nflows.flows import MaskedAutoregressiveFlow as BaseMAF


class MaskedAutoregressiveFlow(BaseMAF):
    """
    Wrapper for MaskedAutoregressiveFlow included in nflows that adds
    additional methods that are used in FlowModel.

    See: https://github.com/bayesiains/nflows/blob/master/nflows/flows/
    autoregressive.py
    """

    def forward(self, x, context=None):
        """
        Apply the forward transformation and return samples in the latent
        space and log |J|
        """
        return self._transform.forward(x)

    def inverse(self, z, context=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log |J|
        """
        return self._transform.inverse(z, context=context)

    def base_distribution_log_prob(self, z, context=None):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow
        """
        return self._distribution.log_prob(z, context=context)

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
