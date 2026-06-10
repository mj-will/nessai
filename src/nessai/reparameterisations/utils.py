# -*- coding: utf-8 -*-
"""
Utilities for handling the reparameterisations.
"""

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..utils.entry_points import get_entry_points
from .base import Reparameterisation
from .null import NullReparameterisation

logger = logging.getLogger(__name__)


class ReparameterisationError(RuntimeError):
    """Exception for reparameterisation errors"""


@dataclass(frozen=True)
class KnownReparameterisation:
    """Dataclass to store the reparameterisation class and keyword arguments"""

    name: str
    class_fn: Reparameterisation
    keyword_arguments: dict[str:Any] = field(default_factory=dict)


@dataclass
class ReparameterisationSpec:
    """Normalised representation of a reparameterisation config spec."""

    source_key: str
    spec_index: int
    reparameterisation: str | None
    source_is_parameter: bool
    parameters: list[str] | None
    kwargs: dict[str, Any] = field(default_factory=dict)


class ReparameterisationDict(dict):
    """Dictionary of reparameterisations

    This dictionary is used to store the known reparameterisations and
    provides a method to add new reparameterisations.
    """

    def add_reparameterisation(self, name, class_fn, keyword_arguments=None):
        """Add a new reparameterisation to the dictionary

        Parameters
        ----------
        name : str
            Name of the reparameterisation.
        class_fn : Reparameterisation
            Reparameterisation class.
        keyword_arguments : dict, optional
            Keyword arguments for the reparameterisation.
        """
        if keyword_arguments is None:
            keyword_arguments = {}
        if name in self:
            raise ValueError(f"Reparameterisation {name} already exists")
        self[name] = KnownReparameterisation(name, class_fn, keyword_arguments)

    def add_external_reparameterisations(self, group):
        entry_points = get_entry_points(group)
        for ep in entry_points.values():
            reparam = ep.load()
            if not isinstance(reparam, KnownReparameterisation):
                raise RuntimeError(
                    f"Invalid external reparameterisation: {reparam}"
                )
            elif reparam.name in self:
                raise ValueError(
                    f"Reparameterisation {reparam.name} already exists"
                )
            logger.debug(f"Adding external reparameterisation: {reparam}")
            self[reparam.name] = reparam


def get_reparameterisation(reparameterisation, defaults=None):
    """Function to get a reparameterisation class from a name

    Parameters
    ----------
    reparameterisation : str, \
            :obj:`nessai.reparameterisations.Reparameterisation`
        Name of the reparameterisations to return or a class that inherits from
        :obj:`~nessai.reparameterisations.Reparameterisation`
    defaults : dict, optional
        Dictionary of known reparameterisations that overrides the defaults.

    Returns
    -------
    :obj:`nessai.reparameteristaions.Reparameterisation`
        Reparameterisation class.
    dict
        Keyword arguments for the specific reparameterisation.
    """
    if defaults is None:
        from . import default_reparameterisations

        defaults = default_reparameterisations

    if isinstance(reparameterisation, str):
        reparam = defaults.get(reparameterisation, None)
        if reparam is None:
            raise ValueError(
                f"Unknown reparameterisation: {reparameterisation}. "
                f"Known reparameterisations are: {list(defaults.keys())}."
            )
        else:
            return reparam.class_fn, copy.deepcopy(reparam.keyword_arguments)
    elif reparameterisation is None:
        return NullReparameterisation, {}
    elif isinstance(reparameterisation, type) and issubclass(
        reparameterisation, Reparameterisation
    ):
        return reparameterisation, {}
    else:
        raise TypeError(
            "Reparameterisation must be a str, None, or class that "
            "inherits from `Reparameterisation`"
        )


def normalise_reparameterisation_spec(
    key: str, cfg: str | dict | list | None, model_names: list[str]
) -> list[dict] | list[str] | list[None]:
    """Normalise a reparameterisation config entry into a list of spec configs.

    Parameters
    ----------
    key : str
        The key of the config entry.
    cfg : str, dict, list, or None
        The config entry to normalise.
    model_names : list of str
        The names of the model parameters.
    """
    if isinstance(cfg, str) or cfg is None:
        return [cfg]
    if isinstance(cfg, dict):
        return [cfg.copy()]
    if isinstance(cfg, list):
        if key in model_names:
            return cfg.copy()
        logger.debug("Assuming list of patterns")
        return [{"parameters": cfg.copy()}]
    raise TypeError(
        f"Unknown config type for: {key}. Expected str, dict or list, "
        f"received instance of {type(cfg)}."
    )


def build_reparameterisation_spec(key, spec_cfg, spec_index, model_names):
    """Build a normalised spec from a single config entry."""
    if key in model_names:
        if isinstance(spec_cfg, str) or spec_cfg is None:
            return ReparameterisationSpec(
                source_key=key,
                spec_index=spec_index,
                reparameterisation=spec_cfg,
                source_is_parameter=True,
                parameters=[key],
            )
        if not isinstance(spec_cfg, dict):
            raise TypeError(
                f"Unknown config type for: {key}. Expected str, dict or list, "
                f"received instance of {type(spec_cfg)}."
            )

        spec_cfg = spec_cfg.copy()
        if spec_cfg.get("reparameterisation", None) is None:
            raise RuntimeError(
                f"No reparameterisation found for {key}. "
                "Check inputs (and their spelling :)). "
                f"Current keys: {list(spec_cfg.keys())}"
            )
        reparameterisation = spec_cfg.pop("reparameterisation")

        if "parameters" in spec_cfg:
            parameters = spec_cfg.pop("parameters")
            if isinstance(parameters, str):
                parameters = [parameters]
            elif parameters is None:
                parameters = []
            else:
                parameters = list(parameters)

            if parameters:
                parameters = list(dict.fromkeys([key, *parameters]))
        else:
            parameters = [key]

        return ReparameterisationSpec(
            source_key=key,
            spec_index=spec_index,
            reparameterisation=reparameterisation,
            source_is_parameter=True,
            parameters=parameters,
            kwargs=spec_cfg,
        )

    if isinstance(spec_cfg, str):
        logger.debug("Assuming reparameterisation name and single parameter")
        spec_cfg = {"parameters": [spec_cfg]}
    elif isinstance(spec_cfg, list):
        logger.debug("Assuming list of patterns")
        spec_cfg = {"parameters": spec_cfg}
    elif not isinstance(spec_cfg, dict):
        raise TypeError(
            f"Unknown config type for: {key}. Expected str or dict, "
            f"received instance of {type(spec_cfg)}."
        )

    spec_cfg = spec_cfg.copy()
    spec_cfg.pop("reparameterisation", None)
    return ReparameterisationSpec(
        source_key=key,
        spec_index=spec_index,
        reparameterisation=key,
        source_is_parameter=False,
        parameters=spec_cfg.pop("parameters", None),
        kwargs=spec_cfg,
    )


def parse_reparameterisations(
    reparameterisations, model_names, class_name=None
):
    """Parse user reparameterisation config into ordered specs."""
    if reparameterisations is None:
        logger.info(
            "No reparameterisations provided, using default "
            f"reparameterisations included in {class_name or 'the proposal class'}"
        )
        reparameterisations = {}
    else:
        reparameterisations = copy.deepcopy(reparameterisations)

    if isinstance(reparameterisations, str):
        reparameterisations = {
            reparameterisations: {"parameters": model_names}
        }
    elif not isinstance(reparameterisations, dict):
        raise TypeError(
            "Reparameterisations must be a dictionary, string or None, "
            f"received {type(reparameterisations).__name__}"
        )

    specs = []
    for key, cfg in reparameterisations.items():
        spec_configs = normalise_reparameterisation_spec(key, cfg, model_names)
        for spec_index, spec_cfg in enumerate(spec_configs):
            specs.append(
                build_reparameterisation_spec(
                    key, spec_cfg, spec_index, model_names
                )
            )
    return specs


def resolve_reparameterisation_parameters(parameters, available_parameters):
    """Resolve parameter names or regex patterns for reparameterisations."""
    if parameters is None:
        return None

    if isinstance(parameters, str):
        patterns = [parameters]
    else:
        patterns = list(parameters)

    known_parameters = list(dict.fromkeys(available_parameters))

    matches = []
    for pattern in patterns:
        if pattern in known_parameters:
            matches.append(pattern)
            continue

        regex = re.compile(pattern)
        pattern_matches = list(filter(regex.fullmatch, known_parameters))
        if pattern_matches:
            matches.extend(pattern_matches)
        else:
            logger.warning(
                f"No matches found for pattern: {pattern}. "
                f"Known parameters are: {known_parameters}"
            )

    return list(dict.fromkeys(matches))
