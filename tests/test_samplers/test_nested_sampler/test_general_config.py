# -*- coding: utf-8 -*-
"""
Test general config functions called when the nested sampler is initialised.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai.samplers.nestedsampler import NestedSampler


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("shrinkage_expectation", ["t", "logt"])
@pytest.mark.parametrize(
    "flow_class, flow_proposal_class, expected_proposal_call",
    [
        (None, None, None),
        (None, "test", "test"),
        ("test_old", "test", "test_old"),
    ],
)
def test_init(
    sampler,
    plot,
    shrinkage_expectation,
    flow_class,
    flow_proposal_class,
    expected_proposal_call,
):
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
            flow_class=flow_class,
            flow_proposal_class=flow_proposal_class,
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
    sampler.configure_flow_proposal.assert_called_once_with(
        expected_proposal_call,
        None,  # flow config is None
        False,  # proposal_plots is False by default
        poolsize=100,
    )


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


@pytest.mark.parametrize("plot", [False, True])
@pytest.mark.parametrize("has_proposal", [False, True])
def test_update_output(sampler, tmp_path, plot, has_proposal):
    output = tmp_path / "new"
    sampler.plot = plot
    if has_proposal:
        sampler._flow_proposal = MagicMock()
        sampler._flow_proposal.output = tmp_path / "orig" / "proposal"
    else:
        sampler._flow_proposal = None
    with patch("nessai.samplers.base.BaseNestedSampler.update_output") as mock:
        NestedSampler.update_output(sampler, output)

    mock.assert_called_once_with(output)
    if has_proposal:
        sampler._flow_proposal.update_output.assert_called_once_with(
            os.path.join(output, "proposal", "")
        )

    if plot:
        assert os.path.exists(os.path.join(output, "diagnostics"))


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


def test_initialise_history(sampler):
    sampler.history = None

    def fn():
        sampler.history = {"acceptance": []}

    with patch(
        "nessai.samplers.nestedsampler.BaseNestedSampler.initialise_history",
        side_effect=fn,
    ) as mock_parent:
        NestedSampler.initialise_history(sampler)

    mock_parent.assert_called_once()
    assert "rolling_p" in sampler.history
