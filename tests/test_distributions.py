# -*- coding: utf-8 -*-
"""
Test the included distributions.
"""

import numpy as np
import pytest

from unittest.mock import MagicMock, create_autospec

from nessai.distributions import (
    CategoricalDistribution,
    InterpolatedDistribution
)


@pytest.fixture
def categorical():
    return create_autospec(CategoricalDistribution)


@pytest.fixture
def interpolated():
    return create_autospec(InterpolatedDistribution)


@pytest.mark.parametrize(
    'n, classes', [(None, None), (None, [1, 2]), (2, None), (2, [1, 2])]
)
@pytest.mark.parametrize('samples', [None, [1, 2]])
def test_categorical_init(categorical, n, classes, samples):
    """Test the init method"""
    categorical.update_samples = MagicMock()
    CategoricalDistribution.__init__(
        categorical, n=n, classes=classes, samples=samples)

    if samples is not None:
        categorical.update_samples.assert_called_once_with(samples)

    assert categorical.n == (2 if (n or classes) else None)
    assert categorical.classes == (sorted(classes) if classes else None)


def test_categorical_update_samples_new(categorical):
    """Test updating the samples in the categorical distribution when
    the distribution has not been initialised.
    """
    categorical.classes = None
    categorical.n = None
    categorical.p = None
    categorical.samples = None
    samples = np.random.choice([1, 2], size=10, p=[0.2, 0.8])
    actual_p = [
        (samples == 1).sum() / 10,
        (samples == 2).sum() / 10
    ]
    CategoricalDistribution.update_samples(categorical, samples)

    assert categorical.n == 2
    np.testing.assert_equal(categorical.classes, [1, 2])
    np.testing.assert_equal(categorical.p, actual_p)


@pytest.mark.parametrize('p', [[0.2, 0.8], [0.0, 1.0]])
def test_categorical_update_samples_existing_reset(categorical, p):
    """Test updating the samples in the categorical distribution when
    the existing samples are discarded.
    """
    categorical.classes = [1, 2]
    categorical.n = 2
    categorical.p = [0.1, 0.9]
    categorical.samples = np.random.choice([1, 2], size=10, p=categorical.p)
    samples = np.random.choice([1, 2], size=10, p=p)
    actual_p = [
        (samples == 1).sum() / 10,
        (samples == 2).sum() / 10
    ]
    CategoricalDistribution.update_samples(categorical, samples, reset=True)

    assert categorical.n == 2
    np.testing.assert_equal(categorical.classes, [1, 2])
    np.testing.assert_equal(categorical.p, actual_p)


@pytest.mark.parametrize('p', [[0.2, 0.8], [0.0, 1.0]])
def test_categorical_update_samples_existing_no_reset(categorical, p):
    """Test updating the samples in the categorical distribution when
    there are existing samples which are not discarded
    """
    categorical.classes = [1, 2]
    categorical.n = 2
    categorical.p = [0.1, 0.9]
    categorical.samples = np.random.choice([1, 2], size=10, p=categorical.p)
    samples = np.random.choice([1, 2], size=10, p=p)
    all_samples = np.concatenate([categorical.samples, samples])
    actual_p = [
        (all_samples == 1).sum() / all_samples.size,
        (all_samples == 2).sum() / all_samples.size
    ]
    CategoricalDistribution.update_samples(categorical, samples, reset=False)

    assert categorical.n == 2
    np.testing.assert_equal(categorical.classes, [1, 2])
    np.testing.assert_equal(categorical.p, actual_p)
    np.testing.assert_equal(categorical.samples, all_samples)


def test_categorical_update_samples_2d(categorical):
    """Assert an error is raised if the samples are 2d dimensional and
    the extra dimnesion is not removed by np.squeeze.
    """
    samples = np.random.choice([1, 2], size=(2, 10))
    with pytest.raises(RuntimeError) as excinfo:
        CategoricalDistribution.update_samples(categorical, samples)
    assert 'Samples must be a 1-dimensional array' in str(excinfo.value)


def test_categorical_update_samples_unknown_class(categorical):
    """Assert an error is raised if the samples contain a different class."""
    samples = [1, 3, 3]
    categorical.classes = [1, 2]
    with pytest.raises(RuntimeError) as excinfo:
        CategoricalDistribution.update_samples(categorical, samples)
    assert 'New samples contain different classes' in str(excinfo.value)


def test_categorical_update_samples_too_many_class(categorical):
    """Assert an error is raised if the samples contain more classes than n.

    This shouldn't happen during regular use but it is possible for the use
    to call the class in such a way that it can happen.
    """
    samples = [1, 2, 3]
    categorical.n = 2
    categorical.classes = None
    with pytest.raises(RuntimeError) as excinfo:
        CategoricalDistribution.update_samples(categorical, samples)
    assert 'Categorical distribution has 2 classes, 3 given.' in \
        str(excinfo.value)


@pytest.mark.parametrize('p', [[0.2, 0.8], [1.0, 0.0]])
def test_categorical_log_prob(categorical, p):
    """Test the log probability.

    For each class it should be log(p), where p is the fraction of the total
    samples.
    """
    c = [1, 2]
    samples = np.array([1, 2])
    categorical.classes = c
    categorical.p = p

    log_prob = CategoricalDistribution.log_prob(categorical, samples)

    assert log_prob[0] == np.log(p[0])
    assert log_prob[1] == np.log(p[1])


def test_categorical_sample(categorical):
    """Test sampling from the categorical distribution"""
    categorical.p = [0.2, 0.8]
    categorical.classes = [1, 2]
    n = 100
    expected_log_prob = 0.5 * np.ones(n)
    categorical.log_prob = MagicMock(return_value=expected_log_prob)
    samples, log_prob = CategoricalDistribution.sample(categorical, n=n)
    classes, counts = np.unique(samples, return_counts=True)

    categorical.log_prob.assert_called_once_with(samples)

    np.testing.assert_equal(classes, [1, 2])
    np.testing.assert_array_equal(log_prob, expected_log_prob)
    assert np.abs(0.2 - counts[0] / n) < np.sqrt(counts[0])
    assert np.abs(0.8 - counts[1] / n) < np.sqrt(counts[1])
