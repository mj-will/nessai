# -*- coding: utf-8 -*-
"""
Test the base flow class
"""
import pytest

from nessai.flows import BaseFlow


@pytest.mark.parametrize('method', ['forward', 'inverse', 'sample', 'log_prob',
                                    'base_distribution_log_prob',
                                    'forward_and_log_prob',
                                    'sample_and_log_prob'])
def test_base_flow_methods(method):
    """Test to make sure all of the method in the base class raise errors"""
    m = getattr(BaseFlow(), method)
    assert m
    with pytest.raises(NotImplementedError):
        m(None)
