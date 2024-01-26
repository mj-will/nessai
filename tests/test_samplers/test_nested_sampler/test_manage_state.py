# -*- coding: utf-8 -*-
"""
Tests related to checking and updating the state of sampler and the history.
"""
import os
import pytest
from unittest.mock import MagicMock, patch

from nessai.livepoint import parameters_to_live_point
from nessai.samplers.nestedsampler import NestedSampler


@pytest.fixture
def sampler(sampler):
    """A sampler mock configured to work with update state"""
    # Stored stats
    sampler.history = dict(
        iterations=[10],
        population_radii=[1.0],
        population_iterations=[0],
        population_acceptance=[0.5],
        likelihood_evaluations=[10],
        min_log_likelihood=[0.0],
        max_log_likelihood=[100.0],
        logZ=[-100.0],
        dlogZ=[100.0],
        mean_acceptance=[0.5],
    )
    # Attributed used to update stats
    sampler.model = MagicMock()
    sampler.model.likelihood_evaluations = 100
    sampler.logLmin = 0.0
    sampler.logLmax = 150.0
    sampler.condition = 50.0

    sampler.state = MagicMock()
    sampler.state.logZ = -50.0
    sampler.state.info = [0.0]
    sampler.state.log_evidence_error = 0.1

    sampler.proposal = MagicMock()
    sampler.proposal.r = 2.0
    sampler.proposal.population_acceptance = 0.4

    sampler.mean_acceptance = 0.5
    sampler.mean_block_acceptance = 0.5
    sampler.block_acceptance = 0.5
    sampler.block_iteration = 5

    sampler.checkpoint = MagicMock()

    return sampler


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("uninformed", [False, True])
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


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("train", [False, True])
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


def test_update_history(sampler):
    """Test updating the history dictionary"""

    sampler.iteration = 15

    with patch(
        "nessai.samplers.nestedsampler.BaseNestedSampler.update_history"
    ) as mock:
        NestedSampler.update_history(sampler)

    mock.assert_called_once()

    assert sampler.history["population_acceptance"] == [0.5]
    assert sampler.history["min_log_likelihood"] == [0.0, 0.0]
    assert sampler.history["max_log_likelihood"] == [100.0, 150.0]
    assert sampler.history["logZ"] == [-100.0, -50.0]
    assert sampler.history["dlogZ"] == [100.0, 50.0]
    assert sampler.history["mean_acceptance"] == [0.5, 0.5]
    assert sampler.history["iterations"] == [10, 15]
    sampler.checkpoint.assert_not_called()


@patch("nessai.samplers.nestedsampler.NestedSampler.checkpoint")
def test_update_state_checked_acceptance(mock, sampler):
    """Test the behaviour of update state if `_checked_population` is False.

    Checks to make sure the correct statistics are stored and the other
    checks are not called because it / nlive and it / (nlive/10) != 0
    """
    sampler.iteration = 11
    sampler.proposal._checked_population = False
    sampler.checkpointing = False
    sampler.update_history = MagicMock()

    NestedSampler.update_state(sampler)

    sampler.update_history.assert_not_called()

    assert sampler.history["population_acceptance"] == [0.5, 0.4]
    assert sampler.history["population_radii"] == [1.0, 2.0]
    assert sampler.history["population_iterations"] == [0, 11]
    assert sampler.proposal._checked_population is True
    mock.assert_not_called()


def test_update_state_history(sampler):
    """Test the behaviour of updated state if it / (nlive/10) == 0

    Checks to make sure the correct statistics are saved and metrics related to
    population are not updated.
    """
    sampler.iteration = 10
    sampler.proposal._checked_population = True
    sampler.checkpointing = False
    sampler.update_history = MagicMock()

    NestedSampler.update_state(sampler)

    sampler.update_history.assert_called_once()

    assert sampler.history["population_acceptance"] == [0.5]
    sampler.checkpoint.assert_not_called()

    assert sampler.proposal.ns_acceptance == 0.5


@pytest.mark.parametrize("plot", [False, True])
@patch("nessai.samplers.nestedsampler.plot_indices")
def test_update_state_every_nlive(mock_plot, plot, sampler):
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
    sampler.output = os.getcwd()
    sampler.insertion_indices = range(2 * sampler.nlive)
    sampler.checkpointing = False
    sampler.update_history = MagicMock()

    NestedSampler.update_state(sampler)

    sampler.update_history.assert_called()
    sampler.check_insertion_indices.assert_called_once()
    assert sampler.block_iteration == 0
    assert sampler.block_acceptance == 0.0
    assert sampler.history["population_acceptance"] == [0.5]

    if plot:
        sampler.plot_state.assert_called_once_with(
            filename=os.path.join(os.getcwd(), "state.png")
        )
        sampler.plot_trace.assert_called_once()
        mock_plot.assert_called_once_with(
            sampler.insertion_indices[-100:],
            100,
            plot_breakdown=False,
            filename=os.path.join(
                os.getcwd(), "diagnostics", "insertion_indices_100.png"
            ),
        )
    else:
        sampler.plot_state.assert_not_called()
        sampler.plot_trace.assert_not_called()
        mock_plot.assert_not_called()


@patch("nessai.samplers.nestedsampler.plot_indices")
def test_update_state_force(mock_plot, sampler):
    """Test the update that happens if force=True.

    Checks that plot_indices is not called even if plotting is enabled.
    """
    sampler.iteration = 111
    sampler.proposal._checked_population = True
    sampler.check_insertion_indices = MagicMock()
    sampler.plot = True
    sampler.plot_state = MagicMock()
    sampler.plot_trace = MagicMock()
    sampler.output = os.getcwd()
    sampler.uninformed_sampling = False
    sampler.checkpointing = False
    sampler.update_history = MagicMock()

    NestedSampler.update_state(sampler, force=True)

    sampler.update_history.assert_called_once()

    mock_plot.assert_not_called()
    sampler.plot_trace.assert_called_once()
    sampler.plot_state.assert_called_once_with(
        filename=os.path.join(os.getcwd(), "state.png")
    )

    assert sampler.history["population_acceptance"] == [0.5]
    assert sampler.block_acceptance == 0.5
    assert sampler.block_iteration == 5


def test_update_state_checkpointing(sampler):
    """Assert the checkpoint function is called"""
    sampler.checkpointing = True
    sampler.checkpoint = MagicMock()
    sampler.iteration = 10
    NestedSampler.update_state(sampler)
    sampler.checkpoint.assert_called_once_with(periodic=True)


def test_update_state_checkpointing_disabled(sampler):
    """Assert the checkpoint function is not called"""
    sampler.checkpointing = False
    sampler.checkpoint = MagicMock()
    sampler.iteration = 10
    NestedSampler.update_state(sampler)
    sampler.checkpoint.assert_not_called()


def test_get_result_dictionary(sampler):
    """Assert the correct dictionary is returned"""
    from datetime import timedelta

    base_result = dict(
        seed=1234,
        history=dict(
            likelihood_evaluations=[10],
            sampling_time=[2],
        ),
    )
    sampler.nlive = 1
    sampler.iteration = 3
    sampler.min_log_likelihood = [-3, -2, 1]
    sampler.max_log_likelihood = [1, 2, 3]
    sampler.likelihood_evaluations = 3
    sampler.logZ_history = [1, 2, 3]
    sampler.mean_acceptance_history = [1, 2, 3]
    sampler.rolling_p = [0.5]
    sampler.population_iterations = []
    sampler.population_acceptance = []
    sampler.training_iterations = []
    sampler.insertion_indices = []
    sampler.training_time = timedelta()
    sampler.proposal_population_time = timedelta()
    sampler.nested_samples = [
        parameters_to_live_point((1, 2), ["x", "y"]),
        parameters_to_live_point((3, 4), ["x", "y"]),
    ]
    sampler.final_p_value = 0.5
    sampler.final_ks_statistic = 0.1
    sampler.state.log_posterior_weights = [0.5, 0.5]

    with patch(
        "nessai.samplers.base.BaseNestedSampler.get_result_dictionary",
        return_value=base_result,
    ) as mock:
        out = NestedSampler.get_result_dictionary(sampler)

    mock.assert_called_once()

    assert out["seed"] == 1234
    assert "history" in out
