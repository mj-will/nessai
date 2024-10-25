from typing import Dict, List, Tuple, Union

import numpy as np

from .base import Reparameterisation


class PeriodicReparameterisation(Reparameterisation):
    """Periodic reparameterisation.

    Based on the periodic reparameterisation in pocomc.
    """

    def __init__(
        self,
        parameters=Union[str, List[str]],
        prior_bounds=Dict,
        fit_midpoint: bool = False,
        midpoint: float = np.pi,
    ):
        super().__init__(parameters, prior_bounds)
        self.fit_midpoint = fit_midpoint
        self.midpoint = midpoint
        self.shift = None
        self.width = {p: np.ptp(self.prior_bounds[p]) for p in self.parameters}

    @staticmethod
    def compute_shift(x, prior_bounds, midpoint):
        angles = 2 * np.pi * (x - prior_bounds[0]) / np.ptp(prior_bounds)
        mean_angle = np.angle(np.mean(np.exp(1j * angles))) % (2 * np.pi)
        delta_angle = ((midpoint - mean_angle) + np.pi) % 2 * np.pi - np.pi
        return delta_angle * np.ptp(prior_bounds) / (2 * np.pi)

    def update(self, x):
        if self.fit_midpoint:
            self.shift = self.compute_shift(
                x, self.prior_bounds[self.parameters[0]], self.midpoint
            )
        else:
            self.shift = {p: 0.0 for p in self.parameters}

    def reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        for p, pp in zip(self.parameters, self.prime_parameters):
            x_prime[pp] = self.prior_bounds[p][0] + (
                (x[p] + self.shift[p] - self.prior_bounds[p][0])
                % self.width[p]
            )
        return x, x_prime, log_j

    def inverse_reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        for p, pp in zip(self.parameters, self.prime_parameters):
            x[p] = self.prior_bounds[p][0] + (
                (x_prime[pp] - self.shift[p] - self.prior_bounds[p][0])
                % self.width[p]
            )
        return x, x_prime, log_j
