# -*- coding: utf-8 -*-
"""
Test the main sampling functions
"""
import logging
import numpy as np
import pytest
from unittest.mock import call, MagicMock

from nessai.livepoint import (
    numpy_array_to_live_points,
    parameters_to_live_point,
)
from nessai.samplers.nestedsampler import NestedSampler
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture
def live_points():
    x = numpy_array_to_live_points(np.arange(4)[:, np.newaxis], names=["x"])
    x["logL"] = np.arange(4)
    return x


@pytest.fixture
def sampler(sampler):
    sampler.state = MagicMock()
    sampler.state.logZ = -4.0
    sampler.state.info = [0.0]
    sampler.state.log_evidence_error = 0.1
    sampler.nlive = 4
    sampler.nested_samples = []
    sampler.iteration = 0
    sampler.condition = np.inf
    sampler.block_iteration = 0
    sampler.logLmax = 5
    sampler.insertion_indices = []
    sampler.accepted = 0
    sampler.rejected = 0
    sampler.block_acceptance = 0.0
    sampler.acceptance_history = []
    sampler.info_enabled = True
    sampler.debug_enabled = True
    sampler.log_on_iteration = True
    return sampler


def test_initialise(sampler):
    """Test the initialise method when being used without resuming"""
    sampler._flow_proposal = MagicMock()
    sampler._uninformed_proposal = MagicMock()
    sampler.populate_live_points = MagicMock()

    sampler._flow_proposal.initialised = False
    sampler._uninformed_proposal.initialised = False
    sampler.live_points = None
    sampler.iteration = 0
    sampler.maximum_uninformed = 10
    sampler.condition = 1.0
    sampler.tolerance = 0.1
    sampler.initialised = False
    sampler.uninformed_sampling = True

    NestedSampler.initialise(sampler)

    sampler._flow_proposal.initialise.assert_called_once()
    sampler._uninformed_proposal.initialise.assert_called_once()
    sampler.populate_live_points.assert_called_once()
    assert sampler.initialised is True
    assert sampler.proposal is sampler._uninformed_proposal


def test_initialise_resume(sampler):
    """Test the initialise method when being used after resuming.

    In this case the live points are not None
    """
    sampler._flow_proposal = MagicMock()
    sampler._uninformed_proposal = MagicMock()
    sampler.populate_live_points = MagicMock()

    sampler._flow_proposal.initialised = False
    sampler._uninformed_proposal.initialised = False
    sampler.live_points = [0.0]
    sampler.iteration = 100
    sampler.maximum_uninformed = 10
    sampler.condition = 1.0
    sampler.tolerance = 0.1
    sampler.initialised = False

    NestedSampler.initialise(sampler)

    sampler._flow_proposal.initialise.assert_called_once()
    sampler._uninformed_proposal.initialise.assert_called_once()
    sampler.populate_live_points.assert_not_called()
    assert sampler.initialised is False
    assert sampler.proposal is sampler._flow_proposal


def test_log_state(sampler, caplog):
    """Assert log state outputs to the logger"""
    caplog.set_level(logging.INFO)
    NestedSampler.log_state(sampler)
    assert "logZ:" in str(caplog.text)


def test_finalise(sampler, live_points):
    """Test the finalise method"""
    sampler.live_points = live_points
    sampler.finalised = False

    NestedSampler.finalise(sampler)

    calls = [
        call(0, nlive=4),
        call(1, nlive=3),
        call(2, nlive=2),
        call(3, nlive=1),
    ]

    sampler.state.increment.assert_has_calls(calls)
    sampler.update_state.assert_called_once_with(force=True)
    sampler.state.finalise.assert_called_once()
    assert_structured_arrays_equal(sampler.nested_samples, [*live_points])
    assert sampler.finalised is True


def test_consume_sample(sampler, live_points):
    """Test the default behaviour of consume sample"""
    sampler.live_points = live_points
    new_sample = np.squeeze(parameters_to_live_point((0.5,), ["x"]))
    new_sample["logL"] = 0.5
    sampler.yield_sample = MagicMock()
    sampler.yield_sample.return_value = iter([(1, new_sample)])

    sampler.insert_live_point = MagicMock(return_value=0)
    sampler.check_state = MagicMock()

    NestedSampler.consume_sample(sampler)

    sampler.insert_live_point.assert_called_once_with(new_sample)
    sampler.check_state.assert_not_called()

    assert_structured_arrays_equal(sampler.nested_samples, [live_points[0]])
    assert sampler.logLmin == 0.0
    assert sampler.accepted == 1
    assert sampler.block_acceptance == 1.0
    assert sampler.acceptance_history == [1.0]
    assert sampler.mean_block_acceptance == 1.0
    assert sampler.insertion_indices == [0]


def test_consume_sample_reject(sampler, live_points):
    """Test the default behaviour of consume sample"""
    sampler.live_points = live_points
    reject_sample = parameters_to_live_point((-0.5,), ["x"])
    reject_sample["logL"] = -0.5
    new_sample = np.squeeze(parameters_to_live_point((0.5,), ["x"]))
    new_sample["logL"] = 0.5
    sampler.yield_sample = MagicMock()
    sampler.yield_sample.return_value = iter(
        [(1, reject_sample), (1, new_sample)]
    )

    sampler.insert_live_point = MagicMock(return_value=0)
    sampler.check_state = MagicMock()

    NestedSampler.consume_sample(sampler)

    sampler.insert_live_point.assert_called_once_with(new_sample)
    sampler.check_state.assert_called_once()

    assert_structured_arrays_equal(sampler.nested_samples, [live_points[0]])
    assert sampler.logLmin == 0.0
    assert sampler.rejected == 1
    assert sampler.accepted == 1
    assert sampler.block_acceptance == 0.5
    assert sampler.acceptance_history == [0.5]
    assert sampler.mean_block_acceptance == 0.5
    assert sampler.insertion_indices == [0]


@pytest.mark.parametrize(
    "config",
    [
        {
            "tolerance": 0.1,
            "condition": 0.01,
            "call_finalise": True,
            "call_while": False,
            "close_pool": True,
        },
        {
            "iteration": 10,
            "max_iteration": 10,
            "call_finalise": False,
            "call_while": True,
            "close_pool": False,
        },
    ],
)
def test_nested_sampling_loop(sampler, config):
    """Test the main nested sampling loop.

    This is hard to test because of while loop.
    """
    sampler.prior_sampling = False
    sampler.initialised = False
    sampler.condition = config.get("condition", 0.5)
    sampler.tolerance = config.get("tolerance", 0.1)
    sampler._close_pool = config.get("close_pool", True)
    sampler.max_iteration = config.get("max_iteration")
    sampler.iteration = config.get("iteration", 0)
    sampler.sampling_time = 0.0
    sampler.training_time = 0.0
    sampler.proposal_population_time = 0.0
    sampler.likelihood_calls = 1
    sampler.nested_samples = [1, 2]

    sampler.close_pool = MagicMock()

    sampler.proposal = MagicMock()
    sampler.proposal.pool = True
    sampler.proposal.logl_eval_time.total_seconds = MagicMock()

    sampler.finalised = False
    sampler.finalise = MagicMock()

    sampler.check_resume = MagicMock()
    sampler.check_insertion_indices = MagicMock()
    sampler.checkpoint = MagicMock()

    logZ, samples = NestedSampler.nested_sampling_loop(sampler)

    assert logZ == sampler.state.logZ
    assert samples.tolist() == [1, 2]

    sampler.initialise.assert_called_once_with(live_points=True)
    sampler.check_resume.assert_called_once()

    # If the code entered the while loop, check the functions where called
    if config.get("call_while", True):
        if config.get("iteration", 0):
            sampler.update_state.call_count == 2
        else:
            sampler.update_state.assert_called_once()
        sampler.consume_sample.assert_called_once()
        sampler.check_state.assert_called_once()
        sampler.periodically_log_state.assert_called_once()
    else:
        sampler.check_state.assert_not_called()
        sampler.consume_sample.assert_not_called()
        sampler.update_state.assert_not_called()
        sampler.periodically_log_state.assert_not_called()

    if config.get("call_finalise"):
        sampler.finalise.assert_called_once()
    else:
        sampler.finalise.assert_not_called()

    sampler.check_insertion_indices.assert_called_once_with(rolling=False)
    sampler.checkpoint.assert_called_once_with(periodic=True, force=True)

    if sampler._close_pool:
        sampler.close_pool.assert_called_once()
    else:
        sampler.close_pool.assert_not_called()


@pytest.mark.parametrize("close_pool", [False, True])
def test_nested_sampling_loop_prior_sampling(sampler, close_pool):
    """Test the nested sampling loop for prior sampling"""
    sampler.initialised = False
    sampler.nested_samples = sampler.model.new_point(10)
    sampler.prior_sampling = True
    sampler._close_pool = close_pool
    sampler.close_pool = MagicMock()
    sampler.finalise = MagicMock()
    sampler.log_evidence = -5.99

    evidence, samples = NestedSampler.nested_sampling_loop(sampler)
    sampler.initialise.assert_called_once_with(live_points=True)
    if close_pool:
        sampler.close_pool.assert_called_once()
    else:
        sampler.close_pool.assert_not_called()
    sampler.finalise.assert_called_once()
    assert_structured_arrays_equal(samples, sampler.nested_samples)
    assert evidence == -5.99
