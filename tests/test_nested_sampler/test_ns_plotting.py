# -*- coding: utf-8 -*-
"""
Test plotting in the nested sampler.
"""
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from nessai.nestedsampler import NestedSampler


@pytest.mark.parametrize('track_gradients', [False, True])
@pytest.mark.parametrize('filename', [None, 'test.png'])
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
    sampler.population_radii = 4 * [1.]
    sampler.checkpoint_iterations = [600]
    sampler.likelihood_evaluations = x
    sampler.state = MagicMock()
    sampler.state.track_gradients = track_gradients
    sampler.state.gradients = np.arange(1050)
    sampler.logZ_history = x
    sampler.dZ_history = x
    sampler.mean_acceptance_history = x
    sampler.rolling_p = np.arange(4)

    if filename is not None:
        sampler.output = tmpdir.mkdir('test_plot_state')
        filename = os.path.join(sampler.output, filename)
    fig = NestedSampler.plot_state(sampler, filename)

    if filename is not None:
        assert os.path.exists(filename)
    else:
        assert fig is not None


@pytest.mark.parametrize('samples', [[], [1, 2, 3]])
@patch('nessai.nestedsampler.plot_trace')
def test_plot_trace(mock_plot, sampler, samples):
    """Test the plot_trace method"""
    sampler.nested_samples = samples
    sampler.state = MagicMock()
    sampler.state.log_vols = [1, 2, 3, 4]
    sampler.output = './'

    NestedSampler.plot_trace(sampler)

    if not len(samples):
        mock_plot.assert_not_called
    else:
        mock_plot.assert_called_once_with([2, 3, 4], samples,
                                          filename='.//trace.png')
