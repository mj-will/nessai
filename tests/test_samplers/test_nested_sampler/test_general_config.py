# -*- coding: utf-8 -*-
"""
Test general config functions called when the nested sampler is initialised.
"""
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from nessai.samplers.nestedsampler import NestedSampler


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("shrinkage_expectation", ["t", "logt"])
def test_init(sampler, plot, shrinkage_expectation):
    """Test the init method"""
    pool = "pool"
    n_pool = 2
    model = MagicMock()
    model.verify_model = MagicMock()
    model.configure_pool = MagicMock()

    sampler.setup_output = MagicMock()
    sampler.configure_flow_reset = MagicMock()
    sampler.configure_flow_proposal = MagicMock()
    sampler.configure_uninformed_proposal = MagicMock

    with patch("nessai.samplers.nestedsampler._NSIntegralState") as mock_state:
        NestedSampler.__init__(
            sampler,
            model,
            nlive=100,
            poolsize=100,
            pool=pool,
            n_pool=n_pool,
            plot=plot,
            shrinkage_expectation=shrinkage_expectation,
        )
    assert sampler.initialised is False
    assert sampler.nlive == 100
    mock_state.assert_called_once_with(
        100,
        track_gradients=plot,
        expectation=shrinkage_expectation,
    )
    model.verify_model.assert_called_once()
    model.configure_pool.assert_called_once_with(pool=pool, n_pool=n_pool)


@pytest.mark.parametrize("plot", [False, True])
def test_setup_output(sampler, tmpdir, plot):
    """Test setting up the output directories"""
    path = str(tmpdir.mkdir("test"))
    sampler.plot = plot
    rf = "test.pkl"
    with patch(
        "nessai.samplers.base.BaseNestedSampler.configure_output"
    ) as mock:
        NestedSampler.configure_output(sampler, path, rf)
    mock.assert_called_once_with(path, rf)
    if plot:
        assert os.path.exists(os.path.join(path, "diagnostics"))


def test_configure_max_iteration(sampler):
    """Test to make sure the maximum iteration is set correctly"""
    NestedSampler.configure_max_iteration(sampler, 10)
    assert sampler.max_iteration == 10


def test_configure_no_max_iteration(sampler):
    """Test to make sure if no max iteration is given it is set to inf"""
    NestedSampler.configure_max_iteration(sampler, None)
    assert sampler.max_iteration == np.inf


def test_training_frequency(sampler):
    """Make sure training frequency is set"""
    NestedSampler.configure_training_frequency(sampler, 100)
    assert sampler.training_frequency == 100


@pytest.mark.parametrize("f", [None, "inf", "None"])
def test_training_frequency_on_empty(sampler, f):
    """Test the values that should give 'train on empty'"""
    NestedSampler.configure_training_frequency(sampler, f)
    assert sampler.training_frequency == np.inf
