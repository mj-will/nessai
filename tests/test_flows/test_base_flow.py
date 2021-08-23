# -*- coding: utf-8 -*-
"""
Test the base flow class
"""
import pytest
from unittest.mock import create_autospec, patch

from nessai.flows import BaseFlow, NFlow
from nflows.transforms import Transform


@pytest.fixture
def flow():
    return create_autospec(BaseFlow)


def test_base_flow_abstract_methods():
    """Assert an error is raised if the methods are not implemented."""
    with pytest.raises(TypeError) as excinfo:
        BaseFlow()
    assert 'instantiate abstract class BaseFlow with abstract method' \
        in str(excinfo.value)


@pytest.mark.parametrize(
    'method',
    ['forward', 'inverse', 'sample', 'log_prob', 'base_distribution_log_prob',
     'forward_and_log_prob', 'sample_and_log_prob']
)
def test_base_flow_methods(method, flow):
    """Test to make sure all of the method in the base class raise errors"""
    with pytest.raises(NotImplementedError):
        getattr(BaseFlow, method)(flow, None)


def test_base_flow_to(flow):
    """Assert the to method works as intended"""
    flow.device = None
    with patch('nessai.flows.base.Module.to') as super_init:
        BaseFlow.to(flow, 'cpu')
    super_init.assert_called_once_with('cpu')
    assert flow.device == 'cpu'


def test_nflow_base_transform():
    """Assert an error is raised if the transform class is invalid."""
    class Test:
        pass

    with pytest.raises(TypeError) as excinfo:
        NFlow(Test(), Test())
    assert 'transform must inherit' in str(excinfo.value)


def test_nflow_base_distribution():
    """Assert an error is raised if the distribution class is invalid."""
    class Test:
        pass

    with pytest.raises(TypeError) as excinfo:
        NFlow(Transform(), Test())
    assert 'distribution must inherit' in str(excinfo.value)
