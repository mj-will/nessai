# -*- coding: utf-8 -*-
"""
Test the distributions for flows include in nessai.
"""

import numpy as np
from scipy import stats
import torch
import pytest

from nessai.flows.distributions import (
    MultivariateNormal,
    SphericalTruncatedNormal,
    UniformNBall
)


@pytest.fixture
def dims():
    """Number of dimensions"""
    return 4


@pytest.fixture
def var():
    """Variance to test"""
    return 2


@pytest.fixture
def dist(dims, var):
    """Instance of MultivariateNormal"""
    return MultivariateNormal([dims], var=var)


@pytest.fixture
def scipy_dist(dims, var):
    mean = np.zeros(dims)
    var = var * np.eye(dims)
    return stats.multivariate_normal(mean=mean, cov=var)


def test_log_prob(dist, scipy_dist, dims):
    """Test the log probability of the multivariate normal"""
    x = np.random.rand(1000, dims)
    log_prob = dist._log_prob(torch.from_numpy(x).float(), None).numpy()
    np.testing.assert_array_almost_equal(log_prob, scipy_dist.logpdf(x))


def test_log_prob_invalid_shape(dist, dims):
    """
    Assert the the log prob raises an error for inputs that are the incorrect
    shape.
    """
    x = np.random.rand(10, dims + 2)
    with pytest.raises(ValueError) as excinfo:
        dist._log_prob(torch.from_numpy(x).float(), None)

    assert 'Expected input of shape' in str(excinfo.value)


@pytest.mark.flaky(run=5)
def test_sample(dist, scipy_dist):
    """
    Test the sample method and check if the resulting samples pass a
    KS test using the scipy distribution
    """
    samples = dist._sample(1000, None).numpy()
    p, _ = stats.kstest(samples, scipy_dist.cdf)
    assert p > 0.05


def test_sample_context(dist):
    """
    Test the sample method with a context that is not None.
    Should raise NotImplementedError
    """
    with pytest.raises(NotImplementedError):
        dist._sample(1000, True)


def test_mean(dist, dims):
    """Test the mean of the multivariate normal"""
    np.testing.assert_array_equal(dist._mean(None).numpy(), np.zeros(dims))


def test_mean_context(dist):
    """
    Test the mean of the multivariate normal with context raises
    a not implemneted error
    """
    with pytest.raises(NotImplementedError):
        dist._mean(True)


def test_nball_log_prob():
    """Test the log-probability of the n-ball"""
    dist = UniformNBall(3)

    x_in = torch.tensor([[0.5, 0.5, 0.5]])

    log_prob = dist.log_prob(x_in)

    assert np.isfinite(log_prob)
    assert log_prob == -np.log((4 / 3) * np.pi)


def test_nball_log_prob_out():
    """Test the log-probability of the n-ball outside of bounds"""
    dist = UniformNBall(3)

    x_in = torch.tensor([[0.99, 0.0, 0.0]])

    log_prob = dist.log_prob(x_in)
    print(log_prob)

    assert np.isfinite(log_prob)
    assert log_prob == -np.log((4 / 3) * np.pi)


@pytest.mark.parametrize('radius', [1, 3, 9])
def test_nball_sample(radius):
    """Test the n-ball with different radii"""
    dist = UniformNBall([3], radius=radius)

    x = dist.sample(10).numpy()
    r = np.sqrt(np.sum(x ** 2, axis=1))
    assert (r <= radius).all()


def test_truncate_normal_log_prob():
    """Test the log-probability of the truncated normal"""

    dist = SphericalTruncatedNormal(1, 1.0)

    x = torch.tensor([[0.5], [0.5]])

    true_log_prob = stats.truncnorm(-1, 1).logpdf(x)
    log_prob = dist.log_prob(x)
    np.testing.assert_almost_equal(log_prob, true_log_prob[:, 0])


@pytest.mark.parametrize('dims', [2, 4, 20])
@pytest.mark.parametrize('r', [0.5, 1.0, 2.0, 5.0, 10.0])
def test_truncate_normal_log_prob_n_dims(dims, r):
    """Test the log-probability of the truncated normal in n dimensions"""

    dist = SphericalTruncatedNormal(dims, r)

    x = torch.randn(10, dims)

    true_log_prob = stats.truncnorm(-r, r).logpdf(x).sum(axis=1)
    true_log_prob[np.linalg.norm(x, axis=1) > r] = -np.inf
    log_prob = dist.log_prob(x)
    np.testing.assert_array_almost_equal(log_prob, true_log_prob)


def test_truncate_normal_log_prob_out_of_bounds():
    """
    Test the log-probability of the truncated normal for a point outside the
    radius
    """

    dist = SphericalTruncatedNormal(2, 1.0)

    x = torch.tensor([[1.5, 0.5]])

    true_log_prob = stats.truncnorm(-1, 1).logpdf(x).sum(axis=1)
    log_prob = dist.log_prob(x)
    np.testing.assert_almost_equal(log_prob, true_log_prob)
