# -*- coding: utf-8 -*-
"""Test the proposal utilities"""
from nessai.gw.proposal import GWFlowProposal
from nessai.proposal.flowproposal import FlowProposal
from nessai.proposal.utils import check_proposal_kwargs
import pytest


def test_class_inheritance():
    """Assert classes that inherit from another class work"""

    class A:
        def __init__(self, a, b=1, c=2) -> None:
            self.a = a
            self.b = b
            self.c = c

    class B(A):
        def __init__(self, a, b=3, d=4, **kwargs) -> None:
            super().__init__(a, b, **kwargs)
            self.d = d

    kwargs = dict(b=10, c=20, d=30)
    kwargs_out = check_proposal_kwargs(B, kwargs)
    assert kwargs_out == dict(b=10, c=20, d=30)


def test_kwargs_all_okay(caplog):
    """Assert the kwargs are returned if they all correspond to the class"""
    caplog.set_level("DEBUG")
    kwargs = dict(poolsize=100, volume_fraction=0.9)
    expected = kwargs.copy()
    out = check_proposal_kwargs(GWFlowProposal, kwargs)
    assert out == expected
    assert "All keyword arguments match the proposal class" in str(caplog.text)


@pytest.mark.parametrize("ProposalClass", [FlowProposal, GWFlowProposal])
def test_remove_kwargs(ProposalClass, caplog):
    """Assert kwargs for a different proposal are removed"""
    kwargs = dict(poolsize=100, volume_fraction=0.9, augment_dims=1)
    expected = kwargs.copy()
    expected.pop("augment_dims")
    out = check_proposal_kwargs(ProposalClass, kwargs)
    assert out == expected
    assert "Removing unused keyword arguments" in str(caplog.text)


def test_check_kwargs_strict():
    """Assert kwargs for a different proposal raise an error if strict=True"""
    kwargs = dict(poolsize=100, volume_fraction=0.9, augment_dims=1)
    with pytest.raises(
        RuntimeError,
        match="Keyword arguments contain unknown keys: {'augment_dims'}",
    ):
        check_proposal_kwargs(FlowProposal, kwargs, strict=True)


def test_check_kwargs_error():
    """Assert an error is raised if a keyword argument is not known."""
    with pytest.raises(
        RuntimeError, match=r"Unknown kwargs for FlowProposal: {'not_a_kwarg'}"
    ):
        check_proposal_kwargs(FlowProposal, dict(not_a_kwarg=None))
