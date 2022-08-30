# -*- coding: utf-8 -*-
"""
Test plotting in the nested sampler.
"""
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from nessai.samplers.nestedsampler import NestedSampler


@pytest.mark.parametrize("track_gradients", [False, True])
@pytest.mark.parametrize("filename", [None, "test.png"])
def test_plot_state(sampler, tmpdir, filename, track_gradients):
    """Test making the state plot"""
    x = np.arange(10)
    sampler.min_likelihood = x
    sampler.max_likelihood = x
    sampler.iteration = 1003
    sampler.training_iterations = [256, 711]
    sampler.train_on_empty = False
    sampler.population_iterations = [256, 500, 711, 800]
    sampler.population_acceptance = 4 * [0.5]
    sampler.population_radii = 4 * [1.0]
    sampler.checkpoint_iterations = [600]
    sampler.likelihood_evaluations = x
    sampler.state = MagicMock()
    sampler.state.log_vols = np.linspace(0, -10, 1050)
    sampler.state.track_gradients = track_gradients
    sampler.state.gradients = np.arange(1050)
    sampler.logZ_history = x
    sampler.dZ_history = x
    sampler.mean_acceptance_history = x
    sampler.rolling_p = np.arange(4)

    if filename is not None:
        sampler.output = tmpdir.mkdir("test_plot_state")
        filename = os.path.join(sampler.output, filename)
    fig = NestedSampler.plot_state(sampler, filename)

    if filename is not None:
        assert os.path.exists(filename)
    else:
        assert fig is not None


@pytest.mark.parametrize("samples", [[], [1, 2, 3]])
@pytest.mark.parametrize("filename", [None, "trace.png"])
@patch("nessai.samplers.nestedsampler.plot_trace", return_value="fig")
def test_plot_trace(mock_plot, sampler, tmpdir, samples, filename):
    """Test the plot_trace method"""
    sampler.nested_samples = samples
    sampler.state = MagicMock()
    sampler.state.log_vols = [1, 2, 3, 4]
    sampler.output = os.getcwd()

    if filename is not None:
        sampler.output = tmpdir.mkdir("test_plot_trace")
        filename = os.path.join(sampler.output, filename)

    fig = NestedSampler.plot_trace(sampler, filename=filename)

    if not len(samples):
        mock_plot.assert_not_called()
        assert fig is None
    else:
        mock_plot.assert_called_once_with(
            [2, 3, 4], samples, filename=filename
        )
        assert fig == "fig"


@pytest.mark.parametrize("filename", [None, "trace.png"])
@patch("nessai.samplers.nestedsampler.plot_indices", return_value="fig")
def test_plot_insertion_indices(mock_plot, sampler, filename):
    """Test plotting the insetion indices"""
    nlive = 10
    indices = list(range(20))
    sampler.nlive = nlive
    sampler.insertion_indices = indices
    fig = NestedSampler.plot_insertion_indices(
        sampler, filename=filename, k=True
    )
    assert fig == "fig"
    mock_plot.assert_called_once_with(
        indices, nlive, filename=filename, k=True
    )
