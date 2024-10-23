# -*- coding: utf-8 -*-
"""Utilities for proposal classes"""

import logging
from copy import deepcopy
from inspect import getmro, signature
from typing import Callable, Union
from warnings import warn

from ..utils.settings import _get_all_kwargs

logger = logging.getLogger(__name__)


def check_proposal_kwargs(ProposalClass, kwargs, strict=False):
    """Check the keyword arguments are correct for a proposal class.

    Removes any keyword arguments that correspond to another class. Raises an
    error if the keyword arguments are unknown. For example if the class is
    FlowProposal, it will remove any keyword arguments that correspond to
    AugmentedFlowProposal but not raise an error.

    Parameters
    ----------
    ProposalClass : Type[FlowProposal]
        Class to check the keyword arguments against.
    kwargs : dict
        Dictionary of keyword arguments.
    strict : bool
        Raise an error if any extra keyword arguments are provided even if they
        correspond to other proposals.

    Returns
    -------
    dict
        Dictionary of updated kwargs.
    """
    proposals = list(available_base_flow_proposal_classes().values()) + list(
        available_external_flow_proposal_classes(load=True).values()
    )

    class_keys = set()
    for cls in getmro(ProposalClass):
        class_keys.update(signature(cls).parameters.keys())
        if cls in proposals:
            proposals.remove(cls)

    keys = set(kwargs.keys())
    kwargs_out = deepcopy(kwargs)

    extra_keys = keys - class_keys

    if not extra_keys:
        logger.debug("All keyword arguments match the proposal class")
        return kwargs_out
    elif strict:
        raise RuntimeError(
            f"Keyword arguments contain unknown keys: {extra_keys}"
        )

    allowed_extra_keys = set()

    for proposal in proposals:
        allowed_extra_keys.update(set(_get_all_kwargs(proposal)))

    invalid_keys = extra_keys - allowed_extra_keys

    if invalid_keys:
        raise RuntimeError(
            f"Unknown kwargs for {ProposalClass.__name__}: {invalid_keys}."
        )
    else:
        logger.warning(
            f"Removing unused keyword arguments ({extra_keys}) from kwargs for"
            f" {ProposalClass.__name__}. These are valid keyword arguments but"
            " correspond to other proposal classes."
        )
        for key in extra_keys:
            kwargs_out.pop(key)
    return kwargs_out


def available_base_flow_proposal_classes():
    from ..experimental.gw.proposal import ClusteringGWFlowProposal
    from ..experimental.proposal.clustering import ClusteringFlowProposal
    from ..experimental.proposal.mcmc import MCMCFlowProposal
    from ..gw.proposal import AugmentedGWFlowProposal, GWFlowProposal
    from .augmented import AugmentedFlowProposal
    from .flowproposal import FlowProposal

    base_proposals = {
        "mcmcflowproposal": MCMCFlowProposal,
        "clusteringgwflowproposal": ClusteringGWFlowProposal,
        "augmentedgwflowproposal": AugmentedGWFlowProposal,
        "gwflowproposal": GWFlowProposal,
        "clusteringflowproposal": ClusteringFlowProposal,
        "augmentedflowproposal": AugmentedFlowProposal,
        "flowproposal": FlowProposal,
    }
    return base_proposals


def available_external_flow_proposal_classes(load: bool = False):
    from ..utils.entry_points import get_entry_points

    external_proposals = get_entry_points("nessai.proposals")
    logger.debug(
        f"Found the following external proposals: {external_proposals.keys()}"
    )
    if load:
        for key in external_proposals:
            external_proposals[key] = external_proposals[key].load()
    return external_proposals


def get_flow_proposal_class(
    proposal_class: Union[str, None, Callable],
) -> Callable:
    """Get the proposal class for the standard nested sampler.

    Can also load proposals from entry points. These take priority of the
    default proposals.

    Parameters
    ----------
    proposal_class : Union[str, Proposal, None]
        The name of the proposal class or the class itself. If not specified,
        defaults to :py:obj:`nessai.proposal.flowpropsoal.FlowProposal`.

    Returns
    -------
    Proposal class
    """
    from .flowproposal import FlowProposal

    base_proposals = available_base_flow_proposal_classes()
    external_proposals = available_external_flow_proposal_classes(load=False)

    if proposal_class is None:
        return FlowProposal
    elif isinstance(proposal_class, str):
        proposal_class = proposal_class.lower()
        if proposal_class in external_proposals:
            logger.info("Using external proposal class")
            return external_proposals[proposal_class].load()
        elif proposal_class in base_proposals:
            return base_proposals[proposal_class]
        else:
            raise ValueError(
                f"Unknown proposal class: {proposal_class}. "
                f"Choose from: {list(base_proposals.keys())} or "
                f"{list(external_proposals.keys())}"
            )
    elif issubclass(proposal_class, FlowProposal):
        return proposal_class
    else:
        raise TypeError(
            "Unknown proposal_class type. Must a str, subclass of "
            f"FlowProposal or None. "
            f"Actual input: {proposal_class} ({type(proposal_class)})."
        )


def get_region_sampler_proposal_class(*args, **kwargs):
    warn(
        (
            "`get_region_sampler_proposal_class` is deprecated in favour of "
            "`get_flow_proposal_class`"
        ),
        FutureWarning,
    )
    return get_flow_proposal_class(*args, **kwargs)
