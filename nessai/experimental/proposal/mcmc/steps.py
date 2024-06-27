from functools import lru_cache
import numpy as np


class Step:

    def __init__(self, ensemble: np.ndarray | None = None) -> None:
        self.update_ensemble(ensemble)

    def step(self, z):
        raise NotImplementedError

    def update_ensemble(self, ensemble):
        self.ensemble = ensemble


class GaussianStep(Step):

    def __init__(
        self,
        ensemble: np.ndarray | None = None,
    ) -> None:
        super().__init__(ensemble=ensemble)

    def step(self, z):
        return z + self.sigma * np.random.randn(*z.shape), np.zeros(z.shape[0])


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

    def __init__(
        self,
        ensemble: np.ndarray | None = None,
        mix_fraction: float = 0.5,
        sigma: float = 1e-4,
    ) -> None:
        self.mix_fraction = mix_fraction
        self.sigma = sigma
        super().__init__(ensemble)

    def step(self, z):
        pairs = _get_nondiagonal_pairs(self.ensemble.shape[0])
        indices = np.random.choice(
            pairs.shape[0], size=z.shape[0], replace=True
        )
        diffs = np.diff(self.ensemble[pairs[indices]], axis=1).squeeze(axis=1)
        # n = x.shape[0]
        mix = np.random.rand(z.shape[0]) < self.mix_fraction
        g0 = 2.38 / np.sqrt(2 * z.shape[1])
        scale = np.ones((z.shape[0], 1))
        scale[mix, :] = g0
        error = self.sigma * np.random.randn(*scale.shape)
        z_new = z + scale * diffs + error
        return z_new, np.zeros(z.shape[0])


KNOWN_STEPS = dict(
    gaussian=GaussianStep,
    diff=DifferentialEvolutionStep,
)
