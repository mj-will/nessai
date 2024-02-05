# -*- coding: utf-8 -*-
"""
Test the properties in NestedSampler
"""
from collections import deque
import time
import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock

from nessai.samplers.nestedsampler import NestedSampler


@pytest.fixture()
def sampler(sampler):
    sampler.state = MagicMock()
    return sampler


def test_log_evidence(sampler):
    """Check evidence is returned"""
    sampler.state.logZ = -2
    assert NestedSampler.log_evidence.__get__(sampler) == -2


def test_log_evidence_error(sampler):
    """Check the log-evidence error is returned"""
    sampler.state.log_evidence_error = 0.1
    assert NestedSampler.log_evidence_error.__get__(sampler) == 0.1


def test_information(sampler):
    """Check most recent information estimate is returned"""
    sampler.state.info = [1, 2, 3]
    assert NestedSampler.information.__get__(sampler) == 3


def test_population_time(sampler):
    """Assert the time is the some of the time for individual proposals"""
    sampler._uninformed_proposal = MagicMock()
    sampler._flow_proposal = MagicMock()
    sampler._uninformed_proposal.population_time = 1
    sampler._flow_proposal.population_time = 2
    assert NestedSampler.proposal_population_time.__get__(sampler) == 3


def test_acceptance(sampler):
    """Test the acceptance calculation"""
    sampler.iteration = 10
    sampler.likelihood_calls = 100
    assert NestedSampler.acceptance.__get__(sampler) == 0.1


def test_current_sampling_time(sampler):
    """Test the current sampling time"""
    sampler.finalised = False
    sampler.sampling_time = datetime.timedelta(seconds=10)
    sampler.sampling_start_time = datetime.datetime.now()
    time.sleep(0.01)
    t = NestedSampler.current_sampling_time.__get__(sampler)
    assert t.total_seconds() > 10.0


def test_current_sampling_time_finalised(sampler):
    """Test the current sampling time if the sampling has been finalised"""
    sampler.finalised = True
    sampler.sampling_time = 10
    assert NestedSampler.current_sampling_time.__get__(sampler) == 10


def test_last_updated(sampler):
    """Assert last training iteration is returned"""
    sampler.history = dict(training_iterations=[10, 20])
    assert NestedSampler.last_updated.__get__(sampler) == 20


def test_last_updated_no_training(sampler):
    """Assert None is return if the flow has not been trained"""
    sampler.history = dict(training_iterations=[])
    assert NestedSampler.last_updated.__get__(sampler) == 0


def test_last_updated_no_history(sampler):
    """Assert None is return if the flow has not been trained"""
    sampler.history = None
    assert NestedSampler.last_updated.__get__(sampler) == 0


def test_mean_acceptance(sampler):
    """Assert the mean is returned"""
    sampler.acceptance_history = [1.0, 2.0, 3.0]
    assert NestedSampler.mean_acceptance.__get__(sampler) == 2.0


def test_mean_acceptance_empty(sampler):
    """Assert nan is returned if no points have been proposed"""
    sampler.acceptance_history = deque(maxlen=10)
    assert np.isnan(NestedSampler.mean_acceptance.__get__(sampler))


def test_posterior_effective_sample_size(sampler):
    """Assert the state property is called"""
    sampler.state.effective_n_posterior_samples = 10
    out = NestedSampler.posterior_effective_sample_size.__get__(sampler)
    assert out == 10


def test_birth_log_likelihoods(sampler):
    """Assert the birth log-likelihoods are correct"""
    sampler.state.logLs = [-np.inf, 1, 2, 3, 4]
    dtype = [("it", "i4")]
    sampler.nested_samples = [
        np.array(
            [
                0,
            ],
            dtype=dtype,
        ),
        np.array(
            [
                1,
            ],
            dtype=dtype,
        ),
        np.array(
            [
                2,
            ],
            dtype=dtype,
        ),
        np.array(
            [
                0,
            ],
            dtype=dtype,
        ),
        np.array(
            [
                0,
            ],
            dtype=dtype,
        ),
        np.array(
            [
                4,
            ],
            dtype=dtype,
        ),
        np.array(
            [
                3,
            ],
            dtype=dtype,
        ),
    ]

    expected = np.array([-np.inf, 1, 2, -np.inf, -np.inf, 4, 3])

    out = NestedSampler.birth_log_likelihoods.__get__(sampler)

    np.testing.assert_array_equal(out, expected)
