# -*- coding: utf-8 -*-
"""
Specific proposal methods for sampling gravitational-wave models.
"""
import logging

from ..proposal import FlowProposal
from ..proposal.augmented import AugmentedFlowProposal

from .reparameterisations import get_gw_reparameterisation

logger = logging.getLogger(__name__)


class GWFlowProposal(FlowProposal):
    """Wrapper for FlowProposal that has defaults for CBC-PE"""

    aliases = {
        "chirp_mass": ("mass", None),
        "mass_ratio": ("mass_ratio", None),
        "ra": ("sky-ra-dec", ["DEC", "dec", "Dec"]),
        "dec": ("sky-ra-dec", ["RA", "ra"]),
        "azimuth": ("sky-az-zen", ["zenith", "zen", "Zen", "Zenith"]),
        "zenith": ("sky-az-zen", ["azimuth", "az", "Az", "Azimuth"]),
        "theta_1": ("angle-sine", None),
        "theta_2": ("angle-sine", None),
        "tilt_1": ("angle-sine", None),
        "tilt_2": ("angle-sine", None),
        "theta_jn": ("angle-sine", None),
        "iota": ("angle-sine", None),
        "phi_jl": ("angle-2pi", None),
        "phi_12": ("angle-2pi", None),
        "phase": ("angle-2pi", None),
        "psi": ("angle-pi", None),
        "geocent_time": ("time", None),
        "time_jitter": ("periodic", None),
        "a_1": ("default", None),
        "a_2": ("default", None),
        "chi_1": ("default", None),
        "chi_2": ("default", None),
        "luminosity_distance": ("distance", None),
    }
    """
    Dictionary of aliases used to determine the default reparameterisations
    for common gravitational-wave parameters.
    """
    use_default_reparameterisations = True
    """
    GW specific reparameterisations will be used by default. This is different
    to the parent class where they are disabled by default.
    """

    def get_reparameterisation(self, reparameterisation):
        """Function to get reparameterisations that checks GW defaults and
        aliases
        """
        return get_gw_reparameterisation(reparameterisation)

    def add_default_reparameterisations(self):
        """
        Add default reparameterisations for parameters that have not been
        specified.
        """
        parameters = [
            n
            for n in self.model.names
            if n not in self._reparameterisation.parameters
        ]
        logger.info(f"Adding default reparameterisations for {parameters}")

        for p in parameters:
            logger.debug(f"Trying to add reparameterisation for {p}")
            if p in self._reparameterisation.parameters:
                logger.debug(f"Parameter {p} is already included")
                continue
            name, extra_params = self.aliases.get(p.lower(), (None, None))
            if name is None:
                logger.debug(f"{p} is not a known GW parameter")
                continue
            if extra_params is not None:
                p = [p] + [ep for ep in extra_params if ep in self.model.names]
            else:
                p = [p]
            prior_bounds = {k: self.model.bounds[k] for k in p}
            reparam, kwargs = get_gw_reparameterisation(name)
            logger.info(
                f"Adding reparameterisation {reparam.__name__} for {p} "
                f"with config: {kwargs}"
            )
            self._reparameterisation.add_reparameterisation(
                reparam(parameters=p, prior_bounds=prior_bounds, **kwargs)
            )


class AugmentedGWFlowProposal(AugmentedFlowProposal, GWFlowProposal):
    """Augmented version of GWFlowProposal.

    See :obj:`~nessai.proposal.augmented.AugmentedFlowProposal` and
    :obj:`~nessai.gw.proposal.GWFlowPropsosal`
    """

    pass
