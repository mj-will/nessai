# -*- coding: utf-8 -*-
"""Utilities for proposal classes"""
from copy import deepcopy
from inspect import getmro, signature
import logging


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
    from ..proposal import AugmentedFlowProposal, FlowProposal
    from ..gw.proposal import AugmentedGWFlowProposal, GWFlowProposal

    proposals = {
        AugmentedFlowProposal,
        AugmentedGWFlowProposal,
        FlowProposal,
        GWFlowProposal,
    }

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
        allowed_extra_keys.update(set(signature(proposal).parameters.keys()))

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


def get_region_sampler_proposal_class(proposal_class):
    """Get the proposal class for the standard region sampler.

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

    if proposal_class is not None:
        if isinstance(proposal_class, str):
            proposal_class = proposal_class.lower()
            if proposal_class == "gwflowproposal":
                from ..gw.proposal import GWFlowProposal as proposal_class
            elif proposal_class == "augmentedgwflowproposal":
                from ..gw.proposal import (
                    AugmentedGWFlowProposal as proposal_class,
                )
            elif proposal_class == "flowproposal":
                proposal_class = FlowProposal
            elif proposal_class == "augmentedflowproposal":
                from ..proposal import AugmentedFlowProposal

                proposal_class = AugmentedFlowProposal
            elif proposal_class == "clusteringflowproposal":
                from ..experimental.proposal.clustering import (
                    ClusteringFlowProposal,
                )

                proposal_class = ClusteringFlowProposal
            elif proposal_class == "clusteringgwflowproposal":
                from ..experimental.gw.proposal import (
                    ClusteringGWFlowProposal,
                )

                proposal_class = ClusteringGWFlowProposal
            else:
                raise ValueError(f"Unknown flow class: {proposal_class}")
        elif not issubclass(proposal_class, FlowProposal):
            raise RuntimeError(
                "Flow class must be string or class that "
                "inherits from FlowProposal"
            )
    else:
        proposal_class = FlowProposal
    return proposal_class
