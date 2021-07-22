# -*- coding: utf-8 -*-
"""
Tests related to checking and updating the state of sampler and the history.
"""
import pytest
from unittest.mock import MagicMock, patch

from nessai.nestedsampler import NestedSampler


@pytest.fixture
def sampler(sampler):
    """A sampler mock configured to work with update state"""
    # Stored stats
    sampler.population_acceptance = [0.5]
    sampler.population_radii = [1.0]
    sampler.population_iterations = [0]
    sampler.population_acceptance = [0.5]

    sampler.likelihood_evaluations = [10]
    sampler.min_likelihood = [0.0]
    sampler.max_likelihood = [100.0]
    sampler.logZ_history = [-100.0]
    sampler.dZ_history = [100.0]
    sampler.mean_acceptance_history = [0.5]
    # Attributed used to update stats
    sampler.model = MagicMock()
    sampler.model.likelihood_evaluations = 100
    sampler.logLmin = 0.0
    sampler.logLmax = 150.0
    sampler.condition = 50.0

    sampler.state = MagicMock()
    sampler.state.logZ = -50.0
    sampler.state.info = [0.0]

    sampler.proposal = MagicMock()
    sampler.proposal.r = 2.0
    sampler.proposal.population_acceptance = 0.4

    sampler.mean_acceptance = 0.5
    sampler.mean_block_acceptance = 0.5
    sampler.block_acceptance = 0.5
    sampler.block_iteration = 5

    sampler.checkpoint = MagicMock()

    return sampler


@pytest.mark.parametrize('switch', [False, True])
@pytest.mark.parametrize('uninformed', [False, True])
def test_check_state_force(sampler, switch, uninformed):
    """Test the behaviour of check_state with force=True.

    Training should always start irrespective of other checks and with
    force=True unless uninformed sampling is being used and the switch=False.
    """
    sampler.uninformed_sampling = uninformed
    sampler.check_proposal_switch = MagicMock(return_value=switch)
    sampler.check_training = MagicMock()
    sampler.train_proposal = MagicMock()

    NestedSampler.check_state(sampler, force=True)

    if uninformed and not switch:
        sampler.train_proposal.assert_not_called()
    else:
        sampler.train_proposal.assert_called_once_with(force=True)

    sampler.check_training.assert_not_called()


@pytest.mark.parametrize('force', [False, True])
@pytest.mark.parametrize('train', [False, True])
def test_check_state_train(sampler, force, train):
    """Test the behaviour of check_state with force=False and `check_training`
    returns True, False, True, True, False, True, or False, False.

    Force is used for the return values for `check_training`
    """
    sampler.uninformed_sampling = False
    sampler.check_proposal_switch = MagicMock()
    sampler.check_training = MagicMock(return_value=(train, force))
    sampler.train_proposal = MagicMock()

    NestedSampler.check_state(sampler, force=False)
    if train or force:
        sampler.check_training.assert_called_once_with()
        sampler.train_proposal.assert_called_once_with(force=force)
    else:
        sampler.check_training.assert_called_once_with()
        sampler.train_proposal.assert_not_called()


@patch('nessai.nestedsampler.NestedSampler.checkpoint')
def test_update_state_checked_acceptance(mock, sampler):
    """Test the behaviour of update state if `_checked_population` is False.

    Checks to make sure the correct statistics are stored and the other
    checks are not called because it / nlive and it / (nlive/10) != 0
    """
    sampler.iteration = 11
    sampler.proposal._checked_population = False

    NestedSampler.update_state(sampler)

    assert sampler.population_acceptance == [0.5, 0.4]
    assert sampler.population_radii == [1.0, 2.0]
    assert sampler.population_iterations == [0, 11]
    assert sampler.proposal._checked_population is True
    assert sampler.likelihood_evaluations == [10]
    assert not mock.called


def test_update_state_history(sampler):
    """Test the behaviour of updated state if it / (nlive/10) == 0

    Checks to make sure the correct statistics are saved and metrics related to
    population are not updated.
    """
    sampler.iteration = 10
    sampler.proposal._checked_population = True

    NestedSampler.update_state(sampler)

    assert sampler.population_acceptance == [0.5]
    assert sampler.likelihood_evaluations == [10, 100]
    assert sampler.min_likelihood == [0.0, 0.0]
    assert sampler.max_likelihood == [100.0, 150.0]
    assert sampler.logZ_history == [-100.0, -50.0]
    assert sampler.dZ_history == [100.0, 50.0]
    assert sampler.mean_acceptance_history == [0.5, 0.5]
    assert not sampler.checkpoint.called

    assert sampler.proposal.ns_acceptance == 0.5


@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.parametrize('plot', [False, True])
@patch('nessai.nestedsampler.plot_indices')
def test_update_state_every_nlive(mock_plot, plot, checkpointing, sampler):
    """Test the update that happens every nlive iterations.

    Tests both with plot=True and plot=False
    """
    sampler.nlive = 100
    sampler.iteration = 100
    sampler.proposal._checked_population = True
    sampler.check_insertion_indices = MagicMock()
    sampler.plot = plot
    sampler.uninformed_sampling = True
    sampler.plot_state = MagicMock()
    sampler.plot_trace = MagicMock()
    sampler.output = './'
    sampler.insertion_indices = range(2 * sampler.nlive)
    sampler.checkpointing = checkpointing

    NestedSampler.update_state(sampler)

    if checkpointing:
        sampler.checkpoint.assert_called_once_with(periodic=True)
    else:
        sampler.checkpoint.assert_not_called()
    sampler.check_insertion_indices.assert_called_once()
    assert sampler.block_iteration == 0
    assert sampler.block_acceptance == 0.
    assert sampler.likelihood_evaluations == [10, 100]
    assert sampler.population_acceptance == [0.5]

    if plot:
        sampler.plot_state.assert_called_once_with(filename='.//state.png')
        sampler.plot_trace.assert_called_once()
        mock_plot.assert_called_once_with(
            sampler.insertion_indices[-100:], 100, plot_breakdown=False,
            filename='.//diagnostics/insertion_indices_100.png')
    else:
        assert not sampler.plot_state.called
        assert not sampler.plot_trace.called
        assert not mock_plot.called


@pytest.mark.parametrize('checkpointing', [False, True])
@patch('nessai.nestedsampler.plot_indices')
def test_update_state_force(mock_plot, checkpointing, sampler):
    """Test the update that happens if force=True.

    Checks that plot_indices is not called even if plotting is enabled.
    """
    sampler.iteration = 111
    sampler.proposal._checked_population = True
    sampler.check_insertion_indices = MagicMock()
    sampler.plot = True
    sampler.plot_state = MagicMock()
    sampler.plot_trace = MagicMock()
    sampler.output = './'
    sampler.uninformed_sampling = False
    sampler.checkpointing = checkpointing

    NestedSampler.update_state(sampler, force=True)

    if checkpointing:
        sampler.checkpoint.assert_called_once_with(periodic=True)
    else:
        sampler.checkpoint.assert_not_called()
    assert not mock_plot.called
    assert not sampler.called
    sampler.plot_trace.assert_called_once()
    sampler.plot_state.assert_called_once_with(filename='.//state.png')

    assert sampler.max_likelihood == [100.0, 150.0]
    assert sampler.population_acceptance == [0.5]
    assert sampler.block_acceptance == 0.5
    assert sampler.block_iteration == 5
