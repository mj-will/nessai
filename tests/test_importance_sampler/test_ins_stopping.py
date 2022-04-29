
# -*- coding: utf-8 -*-
"""
Tests related to the stopping criteria
"""
import itertools
import numpy as np
import pytest

from nessai.importancesampler import ImportanceNestedSampler as Sampler


aliases = dict(
    dZ=['evidence', 'dZ'],
    ratio=['ratio'],
)


@pytest.fixture()
def sampler(sampler):
    sampler.stopping_criterion_aliases = aliases
    return sampler


@pytest.mark.parametrize(
    "stop_any, crierion, tolerance, expected",
    [
        (False, [1.0, 0.5], [2.0, 0.1], False),
        (False, [1.0, 0.05], [2.0, 0.1], True),
        (True, [1.0, 0.5], [2.0, 0.1], True),
        (True, [3.0, 0.5], [2.0, 0.1], False),
    ]
)
def test_reached_tolerance(sampler, stop_any, crierion, tolerance, expected):
    """Assert the correct boolean is returned"""
    sampler._stop_any = stop_any
    sampler.criterion = crierion
    sampler.tolerance = tolerance
    assert Sampler.reached_tolerance.__get__(sampler) is expected


def test_configure_stopping_criteria_single(sampler):
    """
    Assert the tolerance and criteria are correctly for a single criterion.
    """
    Sampler.configure_stopping_criterion(sampler, 'evidence', 0.01, 'any')
    assert sampler.tolerance == [0.01]
    assert sampler.stopping_criterion == ['dZ']
    assert sampler.criterion == [np.inf]


def test_configure_stopping_criteria_multiple(sampler):
    """
    Assert the tolerance and criteria are correctly for multiple criteria.
    """
    Sampler.configure_stopping_criterion(
        sampler, ['evidence', 'ratio'], [0.01, np.e], 'any'
    )
    assert sampler.tolerance == [0.01, np.e]
    assert sampler.stopping_criterion == ['dZ', 'ratio']
    assert sampler.criterion == [np.inf, np.inf]


@pytest.mark.parametrize(
    "check_criteria, expected",
    [("any", True), ("all", False)],
)
def test_configure_stopping_criteria_check_criteria(
    sampler, check_criteria, expected,
):
    """Assert the value for _stop_any is set correctly.

    Should be True for 'any' and False for 'all'
    """
    Sampler.configure_stopping_criterion(sampler, 'dZ', 0.01, check_criteria)
    assert sampler._stop_any is expected


def test_configure_stopping_criteria_check_criteria_invalid(sampler):
    """Assert an error is raised if check_criteria is not any or all."""
    with pytest.raises(ValueError) as excinfo:
        Sampler.configure_stopping_criterion(sampler, 'dZ', 0.01, 'some')
    assert "check_criteria must be any or all" in str(excinfo.value)


@pytest.mark.parametrize(
    "criterion",
    list(itertools.chain.from_iterable(
        Sampler.stopping_criterion_aliases.values()
    ))
)
@pytest.mark.integration_test
def test_all_stopping_criteria(model, tmp_path, criterion):
    """Integration test for all stopping criteria"""
    output = tmp_path / criterion
    output.mkdir()
    sampler = Sampler(
        model, output=output, nlive=100, stopping_criterion=criterion,
        max_iteration=2, min_samples=10,
    )
    sampler.initialise()
    sampler.nested_sampling_loop()
    assert getattr(sampler, sampler.stopping_criterion[0]) is not None
