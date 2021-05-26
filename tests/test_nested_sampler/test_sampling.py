# -*- coding: utf-8 -*-
"""
Test the main sampling functions
"""
import numpy as np
import pytest
from unittest.mock import call, MagicMock

from nessai.livepoint import (
    numpy_array_to_live_points,
    parameters_to_live_point
)
from nessai.nestedsampler import NestedSampler


@pytest.fixture
def live_points():
    x = numpy_array_to_live_points(np.arange(4)[:, np.newaxis], names=['x'])
    x['logL'] = np.arange(4)
    return x


@pytest.fixture
def sampler(sampler):
    sampler.state = MagicMock()
    sampler.state.logZ = -4.0
    sampler.nlive = 4
    sampler.nested_samples = []
    sampler.iteration = 0
    sampler.block_iteration = 0
    sampler.logLmax = 5
    sampler.insertion_indices = []
    return sampler


def test_finalise(sampler, live_points):
    """Test the finalise method"""
    sampler.live_points = live_points
    sampler.finalised = False

    NestedSampler.finalise(sampler)

    calls = [
       call(0, nlive=4),
       call(1, nlive=3),
       call(2, nlive=2),
       call(3, nlive=1),
    ]

    sampler.state.increment.assert_has_calls(calls)
    sampler.update_state.assert_called_once_with(force=True)
    sampler.state.finalise.assert_called_once()
    assert sampler.nested_samples == [*live_points]
    assert sampler.finalised is True


def test_consume_sample(sampler, live_points):
    """Test the defauly behaviour of consume sample"""
    sampler.live_points = live_points
    new_sample = parameters_to_live_point((0.5,), ['x'])
    new_sample['logL'] = 0.5
    sampler.yield_sample = MagicMock()
    sampler.yield_sample.return_value = iter([(1, new_sample)])

    NestedSampler.consume_sample(sampler)
