# -*- coding: utf-8 -*-
"""
Test the base flow class
"""
import pytest
from unittest.mock import create_autospec

from nessai.flows import BaseFlow


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
