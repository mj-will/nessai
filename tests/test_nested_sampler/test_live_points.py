# -*- coding: utf-8 -*-
"""
Test functions related to handling live points
"""
import numpy as np
from unittest.mock import MagicMock


def test_insert_live_point(sampler):
    """Test inserting a live point"""
    sampler.live_points = np.arange(-5, 0, 1.0).view([('logL', 'f8')])
    new_point = np.array(-3.5, dtype=[('logL', 'f8')])
    index = sampler.insert_live_point(new_point)
    assert index == 1


def test_populate_live_points(sampler):
    """Test popluting the live points"""
    sampler.yield_sample = MagicMock(
        return_value=iter(zip(np.ones(sampler.nlive),
                              sampler.model.new_point(sampler.nlive)))
        )
    sampler.populate_live_points()
    assert len(sampler.live_points) == sampler.nlive
