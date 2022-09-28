# -*- coding: utf-8 -*-
"""
Test the functions related to when the flow should be trained or reset and
training itself.
"""
import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock
from nessai.samplers.nestedsampler import NestedSampler


def test_configure_flow_reset_false(sampler):
    """Assert the attributes evaluate to false if the inputs are false"""
    NestedSampler.configure_flow_reset(sampler, False, False, False)
    assert not sampler.reset_weights
    assert not sampler.reset_permutations
    assert not sampler.reset_flow


@pytest.mark.parametrize("weights", [10, 5.0])
@pytest.mark.parametrize("permutations", [10, 5.0])
def test_configure_flow_reset(sampler, weights, permutations):
    """Assert the attributes evaluate to false if the inputs are false"""
    NestedSampler.configure_flow_reset(sampler, weights, permutations, False)
    assert sampler.reset_weights == float(weights)
    assert sampler.reset_permutations == float(permutations)


@pytest.mark.parametrize("flow", [10, 5.0])
def test_configure_flow_reset_flow(sampler, flow):
    """Assert reset_flow overwrites the other values"""
    NestedSampler.configure_flow_reset(sampler, 2, 4, flow)
    assert sampler.reset_flow == float(flow)
    assert sampler.reset_weights == float(flow)
    assert sampler.reset_permutations == float(flow)


def test_configure_flow_reset_error_weights(sampler):
    """Assert an error is raised in the weights input is invalid"""
    with pytest.raises(TypeError) as excinfo:
        NestedSampler.configure_flow_reset(sampler, None, 5, 5)
    assert "weights` must be" in str(excinfo.value)


def test_configure_flow_reset_error_permutations(sampler):
    """Assert an error is raised in the permutations input is invalid"""
    with pytest.raises(TypeError) as excinfo:
        NestedSampler.configure_flow_reset(sampler, 5, None, 5)
    assert "permutations` must be" in str(excinfo.value)


def test_configure_flow_reset_error_flow(sampler):
    """Assert an error is raised if the flow input is invalid"""
    with pytest.raises(TypeError) as excinfo:
        NestedSampler.configure_flow_reset(sampler, 5, 5, None)
    assert "`reset_flow` must be" in str(excinfo.value)


def test_check_training_not_completed_training(sampler):
    """
    Assert the flow is forced to train if training did not complete when
    the sampler was checkpointed.
    """
    sampler.completed_training = False
    train, force = NestedSampler.check_training(sampler)
    assert train is True
    assert force is True


def test_check_training_train_on_empty(sampler):
    """
    Assert the flow is forced to train if training the pool is empty and
    `train_on_empty` is true but the proposal was not in the process of
    popluating.
    """
    sampler.completed_training = True
    sampler.train_on_empty = True
    sampler.proposal = MagicMock()
    sampler.proposal.populated = False
    sampler.proposal.populating = False
    train, force = NestedSampler.check_training(sampler)
    assert train is True
    assert force is True


def test_check_training_acceptance(sampler):
    """
    Assert that training will be true but not forced if the acceptance
    threshold is met and retraining on acceptance is enabled.
    """
    sampler.completed_training = True
    sampler.train_on_empty = True
    sampler.proposal = MagicMock()
    sampler.proposal.populated = True
    sampler.proposal.populating = False
    sampler.acceptance_threshold = 0.1
    sampler.mean_block_acceptance = 0.01
    sampler.retrain_acceptance = True
    train, force = NestedSampler.check_training(sampler)
    assert train is True
    assert force is False


def test_check_training_iteration(sampler):
    """
    Assert that training will be true but not forced if a training iteration
    is reached (n iterations have passed since last updated).
    """
    sampler.completed_training = True
    sampler.train_on_empty = True
    sampler.proposal = MagicMock()
    sampler.proposal.populated = True
    sampler.proposal.populating = False
    sampler.acceptance_threshold = 0.1
    sampler.mean_block_acceptance = 0.2
    sampler.retrain_acceptance = False
    sampler.iteration = 3521
    sampler.last_updated = 2521
    sampler.training_frequency = 1000
    train, force = NestedSampler.check_training(sampler)
    assert train is True
    assert force is False


@pytest.mark.parametrize(
    "config",
    [
        dict(),
        dict(train_on_empty=False, populated=False),
        dict(train_on_empty=True, populated=False, populating=True),
        dict(
            mean_acceptance=0.01,
            acceptance_threshold=0.5,
            retrain_acceptance=False,
        ),
        dict(
            mean_acceptance=0.5,
            acceptance_threshold=0.01,
            retrain_acceptance=True,
        ),
        dict(iteration=800, last_updated=0, training_frequency=801),
    ],
)
def test_check_training_false(sampler, config):
    """
    Test a range of different scenarios that should all not start training.
    """
    sampler.completed_training = True
    sampler.train_on_empty = config.get("train_on_empty", False)
    sampler.proposal = MagicMock()
    sampler.proposal.populated = config.get("populated", False)
    sampler.proposal.populating = config.get("populating", False)
    sampler.acceptance_threshold = config.get("acceptance_threshold", 0.1)
    sampler.mean_block_acceptance = config.get("mean_acceptance", 0.2)
    sampler.retrain_acceptance = config.get("retrain_acceptance", False)
    sampler.iteration = config.get("iteration", 3000)
    sampler.last_updated = config.get("last_updated", 2500)
    sampler.training_frequency = config.get("training_frequency", 1000)
    train, force = NestedSampler.check_training(sampler)
    assert train is False
    assert force is False


@pytest.mark.parametrize("training_count", [10, 100])
def test_check_flow_model_reset_weights(sampler, training_count):
    """Assert flow model only weights are reset"""
    sampler.proposal = MagicMock()
    sampler.proposal.reset_model_weights = MagicMock()
    sampler.reset_acceptance = False
    sampler.reset_weights = 10
    sampler.reset_permutations = 0
    sampler.proposal.training_count = training_count

    NestedSampler.check_flow_model_reset(sampler)

    sampler.proposal.reset_model_weights.assert_called_once_with(
        weights=True,
        permutations=False,
    )


@pytest.mark.parametrize("training_count", [10, 100])
def test_check_flow_model_reset_permutations(sampler, training_count):
    """Assert flow model only permutations are reset"""
    sampler.proposal = MagicMock()
    sampler.proposal.reset_model_weights = MagicMock()
    sampler.reset_acceptance = False
    sampler.reset_weights = 0
    sampler.reset_permutations = 10
    sampler.proposal.training_count = training_count

    NestedSampler.check_flow_model_reset(sampler)

    sampler.proposal.reset_model_weights.assert_called_once_with(
        weights=False, permutations=True
    )


@pytest.mark.parametrize("training_count", [10, 100])
def test_check_flow_model_reset_both(sampler, training_count):
    """Assert flow model only permutations are reset"""
    sampler.proposal = MagicMock()
    sampler.proposal.reset_model_weights = MagicMock()
    sampler.reset_acceptance = False
    sampler.reset_weights = 10
    sampler.reset_permutations = 10
    sampler.proposal.training_count = training_count

    NestedSampler.check_flow_model_reset(sampler)

    sampler.proposal.reset_model_weights.assert_called_once_with(
        weights=True,
        permutations=True,
    )


def test_check_flow_model_reset_acceptance(sampler):
    """
    Assert flow model is reset based on acceptance is reset_acceptance is True.
    """
    sampler.proposal = MagicMock()
    sampler.proposal.reset_model_weights = MagicMock()
    sampler.reset_acceptance = True
    sampler.mean_block_acceptance = 0.1
    sampler.acceptance_threshold = 0.5
    sampler.proposal.training_count = 1

    NestedSampler.check_flow_model_reset(sampler)

    sampler.proposal.reset_model_weights.assert_called_once_with(
        weights=True, permutations=True
    )


def test_check_flow_model_reset_not_trained(sampler):
    """
    Verify that the flow model is not reset if it has never been trained.
    """
    sampler.proposal = MagicMock()
    sampler.proposal.reset_model_weights = MagicMock()
    sampler.proposal.training_count = 0

    NestedSampler.check_flow_model_reset(sampler)

    sampler.proposal.reset_model_weights.assert_not_called()


def test_train_proposal_not_training(sampler):
    """Verify the proposal is not trained it has not 'cooled down'"""
    sampler.proposal = MagicMock()
    sampler.proposal.train = MagicMock()
    sampler.iteration = 100
    sampler.last_updated = 90
    sampler.cooldown = 20
    NestedSampler.train_proposal(sampler, force=False)
    sampler.proposal.train.assert_not_called()


def test_train_proposal(sampler, wait):
    """Verify the proposal is trained"""
    sampler.proposal = MagicMock()
    sampler.proposal.train = MagicMock(side_effect=wait)
    sampler.check_flow_model_reset = MagicMock()
    sampler.checkpoint = MagicMock()
    sampler.iteration = 100
    sampler.last_updated = 90
    sampler.cooldown = 20
    sampler.memory = False
    sampler.training_time = datetime.timedelta()
    sampler.training_iterations = []
    sampler.live_points = np.arange(10)
    sampler.checkpoint_on_training = True
    sampler.block_iteration = 10
    sampler.block_acceptance = 0.5

    NestedSampler.train_proposal(sampler, force=True)

    sampler.check_flow_model_reset.assert_called_once()
    sampler.proposal.train.assert_called_once()
    sampler.checkpoint.assert_called_once_with(periodic=True)

    assert sampler.training_iterations == [100]
    assert sampler.training_time.total_seconds() > 0
    assert sampler.completed_training is True
    assert sampler.block_iteration == 0
    assert sampler.block_acceptance == 0


def test_train_proposal_memory(sampler, wait):
    """Verify the proposal is trained with memory"""
    sampler.proposal = MagicMock()
    sampler.proposal.train = MagicMock(side_effect=wait)
    sampler.check_flow_model_reset = MagicMock()
    sampler.checkpoint = MagicMock()
    sampler.iteration = 100
    sampler.last_updated = 90
    sampler.cooldown = 20
    sampler.memory = 2
    sampler.training_time = datetime.timedelta()
    sampler.training_iterations = []
    sampler.nested_samples = np.arange(5)
    sampler.live_points = np.arange(5, 10)
    sampler.checkpoint_on_training = True
    sampler.block_iteration = 10
    sampler.block_acceptance = 0.5

    NestedSampler.train_proposal(sampler, force=True)

    sampler.check_flow_model_reset.assert_called_once()
    sampler.checkpoint.assert_called_once_with(periodic=True)
    sampler.proposal.train.assert_called_once()

    np.testing.assert_array_equal(
        sampler.proposal.train.call_args[0], np.array([[5, 6, 7, 8, 9, 3, 4]])
    )

    assert sampler.training_iterations == [100]
    assert sampler.training_time.total_seconds() > 0
    assert sampler.completed_training is True
    assert sampler.block_iteration == 0
    assert sampler.block_acceptance == 0
