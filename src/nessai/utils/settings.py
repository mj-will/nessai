"""Utilities for determining the settings available in nessai.

Used for bilby and pycbc-inference.
"""

from inspect import getmro, signature
from typing import Any, Callable, List, Tuple


def _get_kwargs(func: Callable) -> dict[str, Any]:
    return {
        k: v.default
        for k, v in signature(func).parameters.items()
        if v.default is not v.empty
    }


def _get_all_kwargs(callable: Callable) -> dict[str, Any]:
    try:
        parameters = {}
        for cls in getmro(callable):
            parameters.update(_get_kwargs(cls))
    except AttributeError:
        parameters = _get_kwargs(callable)
    return parameters


def _get_standard_methods() -> Tuple[List[Callable], List[Callable]]:
    """Get a list of the methods used by the standard sampler and the run
    method.
    """
    from ..flowsampler import FlowSampler
    from ..proposal.utils import (
        available_base_flow_proposal_classes,
        available_external_flow_proposal_classes,
    )
    from ..samplers import NestedSampler

    methods = (
        list(available_external_flow_proposal_classes(load=True).values())
        + list(available_base_flow_proposal_classes().values())
        + [
            NestedSampler,
            FlowSampler,
        ]
    )
    run_methods = [
        FlowSampler.run_standard_sampler,
    ]
    return methods, run_methods


def _get_importance_methods() -> Tuple[List[Callable], List[Callable]]:
    """Get a list of the methods used by the importance nested sampler and by
    the run method.
    """
    from ..flowsampler import FlowSampler
    from ..proposal.importance import ImportanceFlowProposal
    from ..samplers import ImportanceNestedSampler

    methods = [
        ImportanceFlowProposal,
        ImportanceNestedSampler,
        FlowSampler,
    ]
    run_methods = [
        FlowSampler.run_importance_nested_sampler,
    ]
    return methods, run_methods


def get_all_kwargs(
    importance_nested_sampler: bool = False,
    split_kwargs: bool = False,
) -> dict:
    """Get a dictionary of all possible kwargs and their default values.

    Parameters
    ----------
    importance_nested_sampler
        Indicates whether the importance nested sampler will be used or not.
        Defaults to :code:`False` for backwards compatibility.
    split_kwargs
        If True, the kwards are split into kwargs passed to :code:`FlowSampler`
        and those passes to :code:`FlowSampler.run`. If False, all kwargs are
        return in a single dictionary.

    Returns
    -------
    Dictionary of kwargs and their default values.
    """
    if importance_nested_sampler:
        methods, run_methods = _get_importance_methods()
    else:
        methods, run_methods = _get_standard_methods()

    kwargs = {}
    run_kwargs = {}
    for kwds, methods in zip([kwargs, run_kwargs], [methods, run_methods]):
        for m in methods:
            kwds.update(_get_all_kwargs(m))

    if split_kwargs:
        return kwargs, run_kwargs
    else:
        kwargs.update(run_kwargs)
        return kwargs


def get_run_kwargs_list(importance_nested_sampler: bool = False) -> List[str]:
    """Get a list of kwargs used in the run method

    Parameters
    ----------
    importance_nested_sampler
        Indicates whether the importance nested sampler will be used or not.
        Defaults to :code:`False` for backwards compatibility.

    Returns
    -------
    List of kwargs.
    """
    from ..flowsampler import FlowSampler

    if importance_nested_sampler:
        method = FlowSampler.run_importance_nested_sampler
    else:
        method = FlowSampler.run_standard_sampler
    run_kwargs_list = list(_get_all_kwargs(method).keys())
    return run_kwargs_list
