"""Reparameterisations for discrete variables"""

import numpy as np

from .rescale import RescaleToBounds


class Dequantise(RescaleToBounds):
    """Reparameterisation that adds noise to discrete variables and then
    rescales to the specified bounds.

    Can also optionally apply a sigmoid/logit transform after rescaling.
    See :py:class:`~nessai.reparameterisations.rescale.RescaleToBounds` for
    more details.

    Note that :code:`update_bounds` is disabled by default and its use is not
    recommended with this reparameterisation.

    Parameters
    ----------
    parameters : list[str]
        List of parameters to apply the reparameterisation to
    prior_bounds : dict
        Dictionary of prior bounds
    rescale_bounds : Optional[list]
        Bounds to rescale to. Defaults to [-1, 1].
    update_bounds : bool
        Update the bounds for rescaling during sampling. Can be enabled but
        not recommended.
    post_rescaling : Optional[str]
        Name of the rescaling to apply after rescaling to the specified bounds.
    """

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        rescale_bounds=None,
        update_bounds=False,
        post_rescaling=None,
        rng=None,
    ):
        super().__init__(
            parameters=parameters,
            prior_bounds=prior_bounds,
            prior=None,
            rescale_bounds=rescale_bounds,
            boundary_inversion=None,
            detect_edges=False,
            offset=None,
            update_bounds=update_bounds,
            pre_rescaling=None,
            post_rescaling=post_rescaling,
            rng=rng,
        )

        self.has_pre_rescaling = True
        self.has_prime_prior = False

    def set_bounds(self, prior_bounds):
        super().set_bounds(prior_bounds)
        # The allowed bounds must in in the maximum value + 1
        self.bounds = {
            p: [b[0], b[1] + 1] for p, b in self.prior_bounds.items()
        }

    def pre_rescaling(self, x):
        n = len(x)
        return x + self.rng.random(n), np.zeros(n)

    def pre_rescaling_inv(self, x):
        return np.floor(x), np.zeros(len(x))
