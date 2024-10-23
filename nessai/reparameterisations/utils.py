# -*- coding: utf-8 -*-
"""
Utilities for handling the reparameterisations.
"""

import copy

from .base import Reparameterisation
from .null import NullReparameterisation


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
