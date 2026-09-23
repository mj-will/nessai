"""Proposals that use normalising flows."""

from .flowproposal import FlowProposal
from .mcmc import MiniPCNFlowProposal

__all__ = [
    "FlowProposal",
    "MiniPCNFlowProposal",
]
