# -*- coding: utf-8 -*-
"""
Combined reparameterisation.
"""
import logging

logger = logging.getLogger(__name__)


class CombinedReparameterisation(dict):
    """Class to handle multiple reparameterisations

    Parameters
    ----------
    reparameterisations : list, optional
        List of reparameterisations to add to the combined reparameterisations.
        Further reparameterisations can be added using
        `add_reparameterisations`.
    """

    def __init__(self, reparameterisations=None):
        super().__init__()
        self.reparameterisations = {}
        self.parameters = []
        self.prime_parameters = []
        self.requires = []
        if reparameterisations is not None:
            self.add_reparameterisations(reparameterisations)

    @property
    def has_prime_prior(self):
        """Boolean to check if prime prior can be enabled"""
        return all(r.has_prime_prior for r in self.values())

    @property
    def requires_prime_prior(self):
        """Boolean to check if any of the priors require the prime space"""
        return any(r.requires_prime_prior for r in self.values())

    def _add_reparameterisation(self, reparameterisation):
        requires = reparameterisation.requires
        if requires and (
            requires not in self.parameters
            or requires not in self.prime_parameters
        ):
            raise RuntimeError(
                f"Could not add {reparameterisation}, missing requirement(s): "
                f"{reparameterisation.requires}."
            )

        self[reparameterisation.name] = reparameterisation
        self.parameters += reparameterisation.parameters
        self.prime_parameters += reparameterisation.prime_parameters
        self.requires += reparameterisation.requires

    def add_reparameterisation(self, reparameterisation):
        """Add a reparameterisation"""
        self.add_reparameterisations(reparameterisation)

    def add_reparameterisations(self, reparameterisations):
        """Add multiple reparameterisations

        Parameters
        ----------
        reparameterisations : list of :`obj`:Reparameterisation
            List of reparameterisations to add.
        """
        if not isinstance(reparameterisations, list):
            reparameterisations = [reparameterisations]
        for r in reparameterisations:
            self._add_reparameterisation(r)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space
        to x'-space

        Parameters
        ----------
        x : structured array
            Array
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated
        """
        for r in self.values():
            x, x_prime, log_j = r.reparameterise(x, x_prime, log_j, **kwargs)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space
        to x'-space

        Parameters
        ----------
        x : structured array
            Array
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated
        """
        for r in reversed(list(self.values())):
            x, x_prime, log_j = r.inverse_reparameterise(
                x, x_prime, log_j, **kwargs
            )
        return x, x_prime, log_j

    def update_bounds(self, x):
        """
        Update the bounds used for the reparameterisation
        """
        for r in self.values():
            if hasattr(r, "update_bounds"):
                logger.debug(f"Updating bounds for: {r.name}")
                r.update_bounds(x)

    def reset_inversion(self):
        """
        Reset edges for boundary inversion
        """
        for r in self.values():
            if hasattr(r, "reset_inversion"):
                r.reset_inversion()

    def log_prior(self, x):
        """
        Compute any additional priors for auxiliary parameters
        """
        log_p = 0
        for r in self.values():
            if r.has_prior:
                log_p += r.log_prior(x)
        return log_p

    def x_prime_log_prior(self, x_prime):
        """
        Compute the prior in the prime space
        """
        log_p = 0
        for r in self.values():
            log_p += r.x_prime_log_prior(x_prime)
        return log_p
