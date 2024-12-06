import logging
import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Step:
    def __init__(
        self,
        dims: int,
        ensemble: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.dims = dims
        self.update_ensemble(ensemble)
        self.rng = rng or np.random.default_rng()

    def step(self, z):
        raise NotImplementedError

    def update_ensemble(self, ensemble):
        self.ensemble = ensemble

    def update_stats(self, n_accept, n_reject):
        self.n_accept = n_accept
        self.n_reject = n_reject

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class GaussianStep(Step):
    def __init__(
        self,
        dims: int,
        ensemble: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        scale: Optional[float] = None,
        update_scale: bool = True,
        target_acceptance: float = 0.5,
    ) -> None:
        super().__init__(dims=dims, ensemble=ensemble, rng=rng)
        if scale is None:
            self.scale = 2 / self.dims**0.5
        else:
            self.scale = scale
        self.update_scale = update_scale
        self.target_acceptance = target_acceptance

    def step(self, z):
        return self.rng.normal(z, scale=self.scale, size=z.shape), np.zeros(
            z.shape[0]
        )

    def update_stats(self, n_accept, n_reject):
        super().update_stats(n_accept, n_reject)
        if self.update_scale:
            # This follows the same maths as dynesty
            acceptance = self.n_accept / (self.n_accept + self.n_reject)
            logger.debug(f"Current acceptance: {acceptance}")
            self.scale *= math.exp(
                (acceptance - self.target_acceptance)
                / self.dims
                / self.target_acceptance
            )
            logger.debug(f"New scale: {self.scale}")


@lru_cache(maxsize=1)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix with size n, excluding the diagonal.

    This is direct copy the same function from emcee.
    """
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )
    return pairs


class DifferentialEvolutionStep(Step):
    """Differential evolution step for MCMC proposal.

    Based on the differential evolution proposal from CPnest and the dynesty-
    specific implementation in bilby.

    Parameters
    ----------
    dims : int
        The number of dimensions in the parameter space.
    ensemble : np.ndarray, optional
        The ensemble of live points. Default is None.
    mix_fraction : float, optional
        The fraction of the ensemble to mix. Default is 0.5.
    sigma : float, optional
        The standard deviation of the error term added to the steps.
        Default is 1e-5.
    """

    def __init__(
        self,
        dims: int,
        ensemble: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        mix_fraction: float = 0.5,
        sigma: float = 1e-5,
    ) -> None:
        self.pairs = None
        self.n_pairs = None

        super().__init__(dims=dims, ensemble=ensemble, rng=rng)

        self.g0 = 2.38 / np.sqrt(2 * self.dims)
        self.mix_fraction = mix_fraction
        self.sigma = sigma

    def update_ensemble(self, ensemble):
        super().update_ensemble(ensemble)
        if self.ensemble is not None:
            self.pairs = _get_nondiagonal_pairs(self.ensemble.shape[0])
            self.n_pairs = self.pairs.shape[0]

    def step(self, z):
        nc = z.shape[0]
        indices = self.rng.choice(self.n_pairs, size=nc, replace=True)
        diffs = np.diff(self.ensemble[self.pairs[indices]], axis=1).squeeze(
            axis=1
        )
        mix = self.rng.random(nc) < self.mix_fraction
        scale = np.ones((nc, 1))
        scale[mix, :] = self.g0
        error = self.sigma * self.rng.standard_normal((nc, self.dims))
        z_new = z + scale * diffs + error
        return z_new, np.zeros(nc)


class StretchStep(Step):
    """Stretch step for MCMC proposal.

    Based on the stretch move proposal from Goodman & Weare (2010), as
    implemented in :code:`cpnest` and :code:`LALInference`.

    Parameters
    ----------
    dims : int
        The number of dimensions in the parameter space.
    scale : float, optional
        The scale factor for the stretch move. Default is 2.0.
    ensemble : np.ndarray, optional
        The ensemble of live points. Default is None.
    """

    def __init__(
        self,
        dims: int,
        ensemble: np.ndarray = None,
        rng: Optional[np.random.Generator] = None,
        scale: float = 2.0,
    ):
        super().__init__(dims=dims, ensemble=ensemble, rng=rng)
        self.scale = scale

    def step(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nc = z.shape[0]
        indices = self.rng.choice(len(self.ensemble), size=nc, replace=True)
        a = self.ensemble[indices]
        x = self.rng.uniform(-1, 1, size=nc) * np.log(self.scale)
        scale = np.exp(x)
        z_new = a + (z - a) * scale[:, None]
        return z_new, x * self.dims


KNOWN_STEPS = dict(
    gaussian=GaussianStep,
    diff=DifferentialEvolutionStep,
    stretch=StretchStep,
)
