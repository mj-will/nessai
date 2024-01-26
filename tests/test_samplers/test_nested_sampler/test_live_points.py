# -*- coding: utf-8 -*-
"""
Test functions related to handling live points
"""
import os

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from nessai.samplers.nestedsampler import NestedSampler


def test_insert_live_point(sampler):
    """Test inserting a live point"""
    sampler.live_points = np.arange(-5, 0, 1.0).view([("logL", "f8")])
    new_point = np.array(-3.5, dtype=[("logL", "f8")])
    index = NestedSampler.insert_live_point(sampler, new_point)
    assert index == 1


def test_populate_live_points(sampler):
    """Test populating the live points"""
    samples = sampler.model.new_point(sampler.nlive)
    samples["logP"] = 0.0
    samples["logL"] = 0.0
    sampler.yield_sample = MagicMock(
        return_value=iter(zip(np.ones(sampler.nlive), samples))
    )
    NestedSampler.populate_live_points(sampler)
    assert len(sampler.live_points) == sampler.nlive


def test_populate_live_points_nans(sampler):
    """Test populating the live points with NaN values"""
    new_points = sampler.model.new_point(sampler.nlive + 1)
    new_points["logL"] = 0.0
    new_points["logL"][4] = np.nan
    new_points["logP"] = 0.0
    sampler.yield_sample = MagicMock(
        return_value=iter(zip(np.ones(sampler.nlive + 1), new_points))
    )
    NestedSampler.populate_live_points(sampler)
    assert len(sampler.live_points) == sampler.nlive
    assert not np.isnan(sampler.live_points["logL"]).any()


def test_populate_live_points_none_returned(sampler):
    """Assert that if None is returned by yield sample, it is skipped"""
    new_points = sampler.model.new_point(5)
    new_points["logL"] = 0.0
    new_points["logP"] = 0.0
    new_points = [
        None,
    ] + [*new_points]
    sampler.yield_sample = MagicMock(
        return_value=iter(zip(np.ones(len(new_points)), new_points))
    )
    sampler.nlive = 5
    NestedSampler.populate_live_points(sampler)
    np.testing.assert_array_equal(
        sampler.live_points,
        np.sort(new_points[1:], order="logL"),
    )


@pytest.mark.parametrize("rolling", [False, True])
@patch(
    "nessai.samplers.nestedsampler.compute_indices_ks_test",
    return_value=(0.1, 0.5),
)
def test_insertion_indices(mock_fn, rolling, sampler):
    """Test computing the distribution of insertion indices"""
    sampler.iteration = 100
    sampler.history = dict(rolling_p=[])
    sampler.final_p_value = None
    sampler.final_ks_statistic = None
    sampler.insertion_indices = np.random.randint(
        sampler.nlive, size=2 * sampler.nlive
    )

    NestedSampler.check_insertion_indices(sampler, rolling=rolling)

    if rolling:
        assert len(sampler.history["rolling_p"]) == 1
        np.testing.assert_array_equal(
            mock_fn.call_args_list[0][0][0],
            sampler.insertion_indices[-sampler.nlive :],
        )
        assert sampler.final_p_value is None
    else:
        mock_fn.assert_called_once_with(
            sampler.insertion_indices, sampler.nlive
        )
        assert sampler.final_p_value == 0.5
        assert sampler.final_ks_statistic == 0.1


@patch(
    "nessai.samplers.nestedsampler.compute_indices_ks_test",
    return_value=(0, None),
)
def test_insertion_indices_p_none(mock_fn, sampler):
    """Test computing the distribution of insertion indices if p is None"""
    sampler.rolling_p = []
    sampler.insertion_indices = np.random.randint(
        sampler.nlive, size=2 * sampler.nlive
    )

    NestedSampler.check_insertion_indices(sampler, rolling=True)

    assert len(sampler.rolling_p) == 0


@pytest.mark.parametrize("filename", [None, "file.txt"])
@patch("numpy.savetxt")
@patch(
    "nessai.samplers.nestedsampler.compute_indices_ks_test",
    return_value=(0.1, 0.5),
)
def test_insertion_indices_save(mock_fn, mock_save, filename, sampler):
    """Test saving the insertion indices"""
    sampler.output = os.getcwd()
    sampler.insertion_indices = np.random.randint(
        sampler.nlive, size=2 * sampler.nlive
    )

    NestedSampler.check_insertion_indices(
        sampler, rolling=False, filename=filename
    )

    if filename:
        mock_save.assert_called_once_with(
            os.path.join(os.getcwd(), "file.txt"),
            sampler.insertion_indices,
            newline="\n",
            delimiter=" ",
        )


@patch(
    "nessai.samplers.nestedsampler.compute_indices_ks_test",
    return_value=(0.1, 0.01),
)
def test_insertion_indices_p_value_warning(mock_fn, sampler, caplog):
    """Assert a warning is logged if the p-value is less than 0.05"""
    sampler.insertion_indices = [1, 2, 3]
    NestedSampler.check_insertion_indices(sampler, rolling=False)
    assert "less than 0.05" in str(caplog.text)
