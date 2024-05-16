# -*- coding: utf-8 -*-
"""
Proposal methods for used in the nested sampler.
"""
from .analytic import AnalyticProposal
from .augmented import AugmentedFlowProposal
from .flowproposal import FlowProposal
from .rejection import RejectionProposal

__all__ = [
    "AnalyticProposal",
    "AugmentedFlowProposal",
    "FlowProposal",
    "RejectionProposal",
]
