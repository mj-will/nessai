# -*- coding: utf-8 -*-
"""Tests for the transforms included for the flows"""
from nessai.flows.transforms import LULinear
import pytest
import torch


def test_LULinear_weight_inverse():
    """Test the patched weight inverse method.

    Based on the test included in nflows:
    https://github.com/bayesiains/nflows/blob/master/tests/transforms/lu_test.py#L12
    """
    transform = LULinear(2)
    lower, upper = transform._create_lower_upper()
    weight = lower @ upper
    weight_inverse = torch.inverse(weight)

    out = transform.weight_inverse()

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.shape == torch.Size((2, 2))
    assert torch.equal(out, weight_inverse)


@pytest.mark.cuda
def test_LUlinear_CUDA():
    """Test to verify that CUDA works"""
    transform = LULinear(2)
    transform.to('cuda')
    transform.weight_inverse()
