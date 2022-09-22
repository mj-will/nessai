# -*- coding: utf-8 -*-
"""
Test the base flow class
"""
import pytest
from unittest.mock import MagicMock, create_autospec, patch

from nessai.flows.base import BaseFlow, NFlow
from glasflow.nflows.transforms import Transform
from glasflow.nflows.distributions import Distribution


@pytest.fixture
def flow():
    return create_autospec(BaseFlow)


@pytest.fixture
def nflow():
    nflow = create_autospec(NFlow)
    nflow._transform = MagicMock(spec=Transform)
    nflow._distribution = MagicMock(spec=Distribution)
    return nflow


def test_base_flow_abstract_methods():
    """Assert an error is raised if the methods are not implemented."""
    with pytest.raises(TypeError) as excinfo:
        BaseFlow()
    assert "instantiate abstract class BaseFlow with abstract method" in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    "method",
    [
        "forward",
        "inverse",
        "sample",
        "log_prob",
        "base_distribution_log_prob",
        "forward_and_log_prob",
        "sample_and_log_prob",
    ],
)
def test_base_flow_methods(method, flow):
    """Test to make sure all of the method in the base class raise errors"""
    with pytest.raises(NotImplementedError):
        getattr(BaseFlow, method)(flow, None)


@pytest.mark.parametrize(
    "method",
    [
        "freeze_transform",
        "unfreeze_transform",
    ],
)
def test_base_flow_methods_no_args(method, flow):
    """Assert methods with args and are not implemented raise an error"""
    with pytest.raises(NotImplementedError):
        getattr(BaseFlow, method)(flow)


@pytest.mark.parametrize("method", ["end_iteration", "finalise"])
def test_base_flow_pass_methods(method, flow):
    """Test the methods that don't do anything in the base class."""
    assert getattr(BaseFlow, method)(flow) is None


def test_base_flow_to(flow):
    """Assert the to method works as intended"""
    flow.device = None
    with patch("nessai.flows.base.Module.to") as super_init:
        BaseFlow.to(flow, "cpu")
    super_init.assert_called_once_with("cpu")
    assert flow.device == "cpu"


def test_nflow_base_transform():
    """Assert an error is raised if the transform class is invalid."""

    class Test:
        pass

    with pytest.raises(TypeError) as excinfo:
        NFlow(Test(), Test())
    assert "transform must inherit" in str(excinfo.value)


def test_nflow_base_distribution():
    """Assert an error is raised if the distribution class is invalid."""

    class Test:
        pass

    with pytest.raises(TypeError) as excinfo:
        NFlow(Transform(), Test())
    assert "distribution must inherit" in str(excinfo.value)


def test_nflow_finalise(nflow):
    """Assert the methods are called"""
    nflow._transform.finalise = MagicMock()
    nflow._distribution.finalise = MagicMock()
    NFlow.finalise(nflow)
    nflow._transform.finalise.assert_called_once()
    nflow._distribution.finalise.assert_called_once()


def test_nflow_finalise_not_called(nflow):
    """Assert no error is raised the methods are missing"""
    NFlow.finalise(nflow)


def test_nflow_end_iteration(nflow):
    """Assert the methods are called"""
    nflow._transform.end_iteration = MagicMock()
    nflow._distribution.end_iteration = MagicMock()
    NFlow.end_iteration(nflow)
    nflow._transform.end_iteration.assert_called_once()
    nflow._distribution.end_iteration.assert_called_once()


def test_nflow_end_iteration_not_called(nflow):
    """Assert no error is raised the methods are missing"""
    NFlow.end_iteration(nflow)


def test_nflow_freeze(nflow):
    """Assert the transform is frozen"""
    nflow._transform.requires_grad_ = MagicMock()
    NFlow.freeze_transform(nflow)
    nflow._transform.requires_grad_.assert_called_once_with(False)


def test_nflow_unfreeze(nflow):
    """Assert the transform is unfrozen"""
    nflow._transform.requires_grad_ = MagicMock()
    NFlow.unfreeze_transform(nflow)
    nflow._transform.requires_grad_.assert_called_once_with(True)
