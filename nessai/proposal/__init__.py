# -*- coding: utf-8 -*-
"""
Proposal methods for used in the nested sampler.
"""
from .analytic import AnalyticProposal
from .conditional import ConditionalFlowProposal
from .flowproposal import FlowProposal
from .rejection import RejectionProposal

__all__ = ["AnalyticProposal",
           "ConditionalFlowProposal",
           "FlowProposal",
           "RejectionProposal"
           ]
