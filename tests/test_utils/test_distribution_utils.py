# -*- coding: utf-8 -*-
"""
Test utilities for distributions.
"""
import pytest

from nessai.utils.distributions import get_uniform_distribution


def test_get_uniform_distribution_cpu():
    """
    Test function for getting uniform torch distrbution over n dimensions
    when called on cpu
    """
    dist = get_uniform_distribution(10, 1, 'cpu')
    assert dist.sample().get_device() == -1


@pytest.mark.cuda
def test_get_uniform_distribution_cuda():
    """
    Test function for getting uniform torch distrbution over n dimensions
    when called on CUDA
    """
    dist = get_uniform_distribution(10, 1, device='cuda')
    assert dist.sample().get_device() != -1
