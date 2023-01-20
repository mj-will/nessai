# -*- coding: utf-8 -*-
"""
Utilities for sorting.
"""
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..reparameterisations import Reparameterisation


def sort_reparameterisations(
    reparameterisations: List["Reparameterisation"],
    existing_parameters: Optional[List[str]] = None,
    known_parameters: Optional[List[str]] = None,
    initial_sort: bool = True,
) -> List["Reparameterisation"]:
    """Sort reparameterisations based on their parameters and requirements.

    Parameters
    ----------
    reparameterisations : List[Reparameterisation]
        List of reparameterisations.
    existing_parameters : Optional[List[str]]
        List of parameters that are all included.
    known_parameters : Optional[List[str]]
        List of all known parameters. If not specified it is inferred from the
        list of reparameterisations.
    initial_sort : bool
        Toggle initial sorting by the number of requirements.

    Returns
    -------
    List[Reparameterisation]
        Sorted list of reparameterisations.

    Raises
    ------
    ValueError
        If a required parameter is missing from the known parameters.
    """
    ordered = []
    skipped = []
    parameters = existing_parameters.copy() if existing_parameters else []

    if known_parameters is None:
        known_parameters = parameters.copy()
        for r in reparameterisations:
            known_parameters += r.parameters

    if initial_sort:
        reparameterisations = sorted(
            reparameterisations, key=lambda r: len(r.requires)
        )

    for r in reparameterisations:
        if not r.requires or all([req in parameters for req in r.requires]):
            ordered.append(r)
            parameters += r.parameters
        elif any([req not in known_parameters for req in r.requires]):
            raise ValueError(
                f"{r.name} requires {r.requires} which contains parameters "
                f"that are not known ({known_parameters})"
            )
        else:
            skipped.append(r)

    if skipped:
        ordered += skipped
        return sort_reparameterisations(
            ordered,
            existing_parameters=existing_parameters,
            known_parameters=known_parameters,
            initial_sort=False,
        )
    return ordered
