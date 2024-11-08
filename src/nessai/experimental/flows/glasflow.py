from functools import partial

from glasflow.flows import CouplingNSF, RealNVP

from ...flows.base import BaseFlow

known_flows = {
    "nsf": CouplingNSF,
    "realnvp": RealNVP,
}


class GlasflowWrapper(BaseFlow):
    """Wrapper for glasflow flow classes"""

    def __init__(
        self,
        FlowClass,
        n_inputs,
        n_neurons,
        n_blocks,
        n_layers,
        **kwargs,
    ) -> None:
        super().__init__()

        n_conditional_inputs = kwargs.pop("context_features", None)
        self._flow = FlowClass(
            n_inputs=n_inputs,
            n_transforms=n_blocks,
            n_blocks_per_transform=n_layers,
            n_neurons=n_neurons,
            n_conditional_inputs=n_conditional_inputs,
            **kwargs,
        )

    def forward(self, x, context=None):
        return self._flow.forward(x, conditional=context)

    def inverse(self, z, context=None):
        return self._flow.inverse(z, conditional=context)

    def log_prob(self, x, context=None):
        return self._flow.log_prob(x, conditional=context)

    def sample(self, n, context=None):
        return self._flow.sample(n, conditional=context)

    def forward_and_log_prob(self, x, context=None):
        return self._flow.forward_and_log_prob(x, conditional=context)

    def sample_and_log_prob(self, n, context=None):
        return self._flow.sample_and_log_prob(n, conditional=context)

    def sample_latent_distribution(self, n, context=None):
        if context is not None:
            raise ValueError
        return self._flow.distribution.sample(n)

    def base_distribution_log_prob(self, z, context=None):
        if context is not None:
            raise ValueError("Context must be None")
        return self._flow.base_distribution_log_prob(z)

    def freeze_transform(self):
        self._flow._transform.requires_grad_(False)

    def unfreeze_transform(self):
        self._flow._transform.requires_grad_(True)


def get_glasflow_class(name):
    """Get the class for a glasflow flow.

    Note: the name must start with :code:`glasflow.`
    """
    name = name.lower()
    if "glasflow" not in name:
        raise ValueError("'glasflow' missing from name")
    short_name = name.replace("glasflow-", "")
    if short_name not in known_flows:
        raise ValueError(f"{name} is not a known glasflow flow")
    FlowClass = known_flows.get(short_name)
    return partial(GlasflowWrapper, FlowClass)
