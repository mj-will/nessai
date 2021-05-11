# -*- coding: utf-8 -*-
"""Tests related to general aspects of the proposal"""
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

from nessai.proposal import FlowProposal
from nessai.livepoint import numpy_array_to_live_points


def test_draw_populated(proposal):
    """Test the draw method if the proposal is already populated"""
    proposal.populated = True
    proposal.samples = np.arange(3)
    proposal.indices = list(range(3))
    out = FlowProposal.draw(proposal, None)
    assert out == proposal.samples[2]
    assert proposal.indices == [0, 1]


def test_draw_populated_last_sample(proposal):
    """Test the draw method if the proposal is already populated but there
    is only one sample left.
    """
    proposal.populated = True
    proposal.samples = np.arange(3)
    proposal.indices = [0]
    out = FlowProposal.draw(proposal, None)
    assert out == proposal.samples[0]
    assert proposal.indices == []
    assert proposal.populated is False


@pytest.mark.parametrize('update', [False, True])
def test_draw_not_popluated(proposal, update):
    """Test the draw method when the proposal is not populated"""
    import datetime
    proposal.populated = False
    proposal.poolsize = 100
    proposal.population_time = datetime.timedelta()
    proposal.samples = None
    proposal.indices = []
    proposal.update_poolsize = update
    proposal.update_poolsize_scale = MagicMock()
    proposal.ns_acceptance = 0.5

    def mock_populate(*args, **kwargs):
        proposal.populated = True
        proposal.samples = np.arange(3)
        proposal.indices = list(range(3))

    proposal.populate = MagicMock(side_effect=mock_populate)

    out = FlowProposal.draw(proposal, 1.)

    assert out == 2
    assert proposal.populated is True
    assert proposal.population_time.total_seconds() > 0.

    proposal.populate.assert_called_once_with(1., N=100)

    assert proposal.update_poolsize_scale.called == update


@pytest.mark.parametrize('plot', [False, 'all'])
def test_training_plots(proposal, tmpdir, plot):
    """Make sure traings plots are correctly produced"""
    proposal._plot_training = plot
    output = tmpdir.mkdir('test/')

    names = ['x', 'y']
    prime_names = ['x_prime', 'y_prime']
    z = np.random.randn(10, 2)
    x = np.random.randn(10, 2)
    x_prime = x / 2
    proposal.training_data = numpy_array_to_live_points(x, names)
    proposal.training_data_prime = \
        numpy_array_to_live_points(x_prime, prime_names)
    x_gen = numpy_array_to_live_points(x, names)
    x_prime_gen = numpy_array_to_live_points(x_prime, prime_names)
    proposal.dims = 2
    proposal.rescale_parameters = names
    proposal.rescaled_names = prime_names

    proposal.forward_pass = MagicMock(return_value=(z, None))
    proposal.backward_pass = MagicMock(return_value=(x_prime_gen, np.ones(10)))
    proposal.inverse_rescale = MagicMock(return_value=(x_gen, np.ones(10)))
    proposal.check_prior_bounds = lambda *args: args
    proposal.model = MagicMock()
    proposal.model.names = names

    FlowProposal._plot_training_data(proposal, output)

    assert os.path.exists(f'{output}/x_samples.png') is bool(plot)
    assert os.path.exists(f'{output}/x_generated.png') is bool(plot)
    assert os.path.exists(f'{output}/x_prime_samples.png') is bool(plot)
    assert os.path.exists(f'{output}/x_prime_generated.png') is bool(plot)


def test_pool_plot_all(proposal, tmpdir):
    """Test for the plots that show the pool of samples"""
    output = tmpdir.mkdir('test_pool_plot')
    proposal.output = output
    proposal._plot_pool = 'all'
    proposal.populated_count = 0
    x = numpy_array_to_live_points(np.random.randn(10, 2), ['x', 'y'])
    FlowProposal.plot_pool(proposal, None, x)
    assert os.path.exists(f'{output}/pool_0.png')
