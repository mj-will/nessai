# -*- coding: utf-8 -*-
"""
Utilities for sorting.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..reparameterisations import Reparameterisation


def sort_reparameterisations(
    reparameterisations: List["Reparameterisation"],
    existing_parameters: Optional[List[str]] = None,
    existing_prime_parameters: Optional[List[str]] = None,
    known_parameters: Optional[List[str]] = None,
    known_prime_parameters: Optional[List[str]] = None,
    initial_sort: bool = True,
) -> List["Reparameterisation"]:
    """Sort reparameterisations based on their parameters and requirements.

    Parameters
    ----------
    reparameterisations : List[Reparameterisation]
        List of reparameterisations.
    existing_parameters : Optional[List[str]]
        List of parameters that are all included.
    existing_prime_parameters : Optional[List[str]]
        List of prime parameters that are all included.
    known_parameters : Optional[List[str]]
        List of all known parameters. If not specified it is inferred from the
        list of reparameterisations.
    known_prime_parameters : Optional[List[str]]
        List of all known prime parameters. If not specified it is inferred from
        the list of reparameterisations.
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
    prime_parameters = (
        existing_prime_parameters.copy() if existing_prime_parameters else []
    )

    if known_parameters is None:
        known_parameters = parameters.copy()
        for r in reparameterisations:
            known_parameters += r.auxiliary_parameters
        known_parameters = list(dict.fromkeys(known_parameters))

    if known_prime_parameters is None:
        known_prime_parameters = prime_parameters.copy()
        for r in reparameterisations:
            known_prime_parameters += r.output_parameters
        known_prime_parameters = list(dict.fromkeys(known_prime_parameters))

    if initial_sort:
        reparameterisations = sorted(
            reparameterisations,
            key=lambda r: len(r.input_parameters),
        )

    for r in reparameterisations:
        missing_inputs = r.resolve_forward_input_spaces(
            parameters, prime_parameters
        )
        if not missing_inputs:
            ordered.append(r)
            parameters += [
                p for p in r.x_output_parameters if p not in parameters
            ]
            prime_parameters += [
                p for p in r.output_parameters if p not in prime_parameters
            ]
        elif any(
            [
                req not in known_parameters
                and req not in known_prime_parameters
                for req in missing_inputs
            ]
        ):
            raise ValueError(
                f"{r.name} requires inputs {missing_inputs} which are not "
                f"known (x: {known_parameters}, x': {known_prime_parameters})"
            )
        else:
            skipped.append(r)

    if skipped:
        if not ordered:
            raise ValueError(
                "Could not sort reparameterisations, check initial "
                "parameters and dependencies. Current parameters: "
                f"{parameters}. Current prime parameters: {prime_parameters}"
            )
        ordered += skipped
        return sort_reparameterisations(
            ordered,
            existing_parameters=existing_parameters,
            existing_prime_parameters=existing_prime_parameters,
            known_parameters=known_parameters,
            known_prime_parameters=known_prime_parameters,
            initial_sort=False,
        )
    return ordered
