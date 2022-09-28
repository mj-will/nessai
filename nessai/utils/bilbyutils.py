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
        FlowSampler.run,
        AugmentedFlowProposal,
        FlowProposal,
        NestedSampler,
        FlowSampler,
    ]
    return methods


def get_all_kwargs() -> dict:
    """Get a dictionary of all possible kwargs and their default values.

    Returns
    -------
    Dictionary of kwargs and their default values.
    """
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


def get_run_kwargs_list() -> List[str]:
    """Get a list of kwargs used in the run method"""
    from ..flowsampler import FlowSampler

    method = FlowSampler.run

    run_kwargs_list = [
        k
        for k, v in signature(method).parameters.items()
        if v.default is not v.empty
    ]
    return run_kwargs_list
