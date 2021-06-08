
import numpy as np
from scipy import stats
import torch
import pytest

from nessai.flows.distributions import MultivariateNormal


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
