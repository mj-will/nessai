# -*- coding: utf-8 -*-
"""
Utilities to make interfacing with bilby easier.
"""
from inspect import signature
from typing import List, Callable


def _get_standard_methods() -> List[Callable]:
    """Get a list of the methods used by the standard sampler"""
    from ..flowsampler import FlowSampler
    from ..proposal import AugmentedFlowProposal, FlowProposal
    from ..samplers import NestedSampler

    methods = [
        FlowSampler.run_standard_sampler,
        AugmentedFlowProposal,
        FlowProposal,
        NestedSampler,
        FlowSampler,
    ]
    return methods


def _get_importance_methods() -> list:
    """Get a list of the methods used by the importance nested sampler"""
    from ..flowsampler import FlowSampler
    from ..proposal.importance import ImportanceFlowProposal
    from ..samplers import ImportanceNestedSampler

    methods = [
        FlowSampler.run_importance_nested_sampler,
        ImportanceFlowProposal,
        ImportanceNestedSampler,
        FlowSampler,
    ]
    return methods


def get_all_kwargs(importance_nested_sampler: bool = False) -> dict:
    """Get a dictionary of all possible kwargs and their default values.

    Parameters
    ----------
    importance_nested_sampler
        Indicates whether the importance nested sampler will be used or not.
        Defaults to :code:`False` for backwards compatibility.

    Returns
    -------
    Dictionary of kwargs and their default values.
    """
    if importance_nested_sampler:
        methods = _get_importance_methods()
    else:
        methods = _get_standard_methods()

    kwargs = {}
    for m in methods:
        kwargs.update(
            {
                k: v.default
                for k, v in signature(m).parameters.items()
                if v.default is not v.empty
            }
        )
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
    run_kwargs_list = [
        k
        for k, v in signature(method).parameters.items()
        if v.default is not v.empty
    ]
    return run_kwargs_list
