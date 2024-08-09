# -*- coding: utf-8 -*-
"""Test the proposal utilities"""
from nessai.proposal.utils import (
    check_proposal_kwargs,
    get_flow_proposal_class,
)
from nessai.proposal import (
    AugmentedFlowProposal,
    FlowProposal,
)
from nessai.gw.proposal import (
    AugmentedGWFlowProposal,
    GWFlowProposal,
)
from nessai.experimental.proposal.clustering import ClusteringFlowProposal
from nessai.experimental.gw.proposal import ClusteringGWFlowProposal
import pytest
from unittest.mock import MagicMock, patch


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


def test_get_region_sampler_class_none():
    """Test the default flow class"""
    assert get_flow_proposal_class(None) is FlowProposal


@pytest.mark.parametrize(
    "proposal_str, ProposalClass",
    [
        ["FlowProposal", FlowProposal],
        ["AugmentedFlowProposal", AugmentedFlowProposal],
        ["GWFlowProposal", GWFlowProposal],
        ["AugmentedGWFlowProposal", AugmentedGWFlowProposal],
        ["flowproposal", FlowProposal],
        ["augmentedflowproposal", AugmentedFlowProposal],
        ["gwflowproposal", GWFlowProposal],
        ["augmentedgwflowproposal", AugmentedGWFlowProposal],
        ["clusteringflowproposal", ClusteringFlowProposal],
        ["clusteringgwflowproposal", ClusteringGWFlowProposal],
    ],
)
def test_get_flow_proposal_class_str(proposal_str, ProposalClass):
    """Test the correct class is returned"""
    with patch("nessai.utils.entry_points.get_entry_points", return_value={}):
        assert get_flow_proposal_class(proposal_str) is ProposalClass


def test_get_flow_proposal_class_external():
    # Mock class
    ExternalClass = MagicMock()

    # Mock what is normally returned by the entry point before they are loaded
    EntryPointClass = MagicMock()
    EntryPointClass.load = MagicMock(return_value=ExternalClass)

    with patch(
        "nessai.utils.entry_points.get_entry_points",
        return_value={"external_class": EntryPointClass},
    ) as mock_get_entry_points:
        ProposalClass = get_flow_proposal_class("external_class")

    mock_get_entry_points.assert_called_once_with("nessai.proposals")
    EntryPointClass.load.assert_called_once()
    assert ProposalClass is ExternalClass


def test_get_flow_proposal_class_invalid_str():
    """Test to check the error raised if an unknown class is used"""
    with pytest.raises(ValueError, match=r"Unknown proposal class: "):
        get_flow_proposal_class("not_a_valid_class")


def test_get_flow_proposal_class_not_a_subclass():
    """
    Test to check an error is raised in the class does not inherit from
    FlowProposal
    """

    class FakeProposal:
        pass

    with pytest.raises(TypeError, match=r"Unknown proposal_class"):
        get_flow_proposal_class(FakeProposal)
