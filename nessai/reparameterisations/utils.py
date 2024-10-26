# -*- coding: utf-8 -*-
"""
Utilities for handling the reparameterisations.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from ..utils.entry_points import get_entry_points
from .base import Reparameterisation
from .null import NullReparameterisation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnownReparameterisation:
    """Dataclass to store the reparameterisation class and keyword arguments"""

    name: str
    class_fn: Reparameterisation
    keyword_arguments: dict[str:Any] = field(default_factory=dict)


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
