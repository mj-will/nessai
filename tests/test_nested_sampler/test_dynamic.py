# -*- coding: utf-8 -*-
"""
Test the dynamic nested sampler.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.dynamic import DynamicNestedSampler


@pytest.fixture
def sampler():
    return create_autospec(DynamicNestedSampler)


def test_determine_bounds_last_point():

    sampler.nested_samples = np.arange(11)
    weights = np.arange(11)
    print(weights)
    start, end = DynamicNestedSampler.determine_update_bounds(
        sampler, weights, fraction=0.9)

    assert start == 9
    assert end == 10
