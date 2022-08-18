# -*- coding: utf-8 -*-
"""
Test the functions related to yielding new samples
"""
import pytest
from unittest.mock import MagicMock

from nessai.livepoint import parameters_to_live_point
from nessai.samplers.nestedsampler import NestedSampler


@pytest.fixture
def old_sample(model):
    s = parameters_to_live_point([5, 5], names=model.names)
    s["logL"] = 0.0
    s["logP"] = 0.0
    return s


@pytest.fixture
def sampler(sampler, old_sample):
    sampler.proposal = MagicMock()
    sampler.logLmin = sampler.model.log_likelihood(old_sample)
    sampler.logLmax = sampler.logLmin
    return sampler


def test_yield_sample_accept(sampler, old_sample):
    """Test function when sample is accepted"""
    new_sample = parameters_to_live_point([1, 1], names=sampler.model.names)
    new_sample["logL"] = 0.0
    new_sample["logP"] = 0.0
    sampler.proposal.draw = MagicMock(return_value=new_sample)

    count, next_sample = next(NestedSampler.yield_sample(sampler, old_sample))

    assert count == 1
    sampler.proposal.draw.assert_called_once_with(old_sample)
    assert next_sample == new_sample
    assert sampler.logLmax == sampler.model.log_likelihood(new_sample)


def test_yield_sample_reject(sampler, old_sample):
    """Test function when sample is rejected and a new sample is drawn"""
    new_samples = [
        parameters_to_live_point([6, 6], names=sampler.model.names),
        parameters_to_live_point([1, 1], names=sampler.model.names),
    ]
    for s in new_samples:
        s["logL"] = 0.0
        s["logP"] = 0.0

    sampler.proposal.draw = MagicMock(side_effect=new_samples)
    sampler.proposal.populated = True

    count, next_sample = next(NestedSampler.yield_sample(sampler, old_sample))

    assert count == 2
    assert next_sample == new_samples[1]
    sampler.proposal.draw.assert_called_with(old_sample)


def test_yield_sample_not_populated(sampler, old_sample):
    """Test function when sample is rejected and a new sample is not drawn"""
    new_sample = parameters_to_live_point([6, 6], names=sampler.model.names)

    sampler.proposal.draw = MagicMock(return_value=new_sample)
    sampler.proposal.populated = False

    count, next_sample = next(NestedSampler.yield_sample(sampler, old_sample))

    assert count == 1
    assert old_sample == next_sample
    sampler.proposal.draw.assert_called_once_with(old_sample)
