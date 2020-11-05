
import numpy as np
from scipy import stats
import pytest

import nessai.utils as utils

from conftest import cuda


@pytest.mark.parametrize("x, y, log_J", [(0., -np.inf, np.inf),
                                         (1., np.inf, np.inf)])
def test_logit_bounds(x, y, log_J):
    """
    Test logit at the bounds
    """
    with pytest.warns(RuntimeWarning):
        assert utils.logit(x) == (y, log_J)


@pytest.mark.parametrize("x, y, log_J", [(np.inf, 1, -np.inf),
                                         (-np.inf, 0, -np.inf)])
def test_sigmoid_bounds(x, y, log_J):
    """
    Test sigmoid for inf
    """
    assert utils.sigmoid(x) == (y, log_J)


@pytest.mark.parametrize("p", [1e-5, 0.5, 1 - 1e-5])
def test_logit_sigmoid(p):
    """
    Test invertibility of sigmoid(logit(x))
    """
    x = utils.logit(p)
    y = utils.sigmoid(x[0])
    np.testing.assert_equal(p, y[0])
    np.testing.assert_almost_equal(x[1] + y[1], 0)


@pytest.mark.parametrize("p", [-10, -1, 0, 1, 10])
def test_sigmoid_logit(p):
    """
    Test invertibility of logit(sigmoid(x))
    """
    x = utils.sigmoid(p)
    y = utils.logit(x[0])
    np.testing.assert_almost_equal(p, y[0])
    np.testing.assert_almost_equal(x[1] + y[1], 0)


def test_replace_in_list():
    """
    Test if the list produced contains the correct entries in the correct
    locations
    """
    x = [1, 2, 3]
    utils.replace_in_list(x, [1, 2], [5, 4])
    assert x == [5, 4, 3]


def test_replace_in_list_item():
    """
    Test if items are correctly converted to lists in replace_in_list function
    """
    x = [1, 2, 3]
    utils.replace_in_list(x, 3, 4)
    assert x == [1, 2, 4]


@pytest.mark.parametrize("mode", ['D+', 'D-'])
def test_ks_test(mode):
    """
    Test KS test for insertion indices with a specifed mode
    """
    indices = np.random.randint(0, 1000, 1000)
    out = utils.compute_indices_ks_test(indices, 1000, mode=mode)
    assert all([o > 0. for o in out])


def test_ks_test_undefined_mode():
    """
    Test KS test for insertion indices with undefined mode
    """
    indices = np.random.randint(0, 1000, 1000)
    with pytest.raises(RuntimeError):
        utils.compute_indices_ks_test(indices, 1000, mode='two-sided')


def test_ks_test_empty_indices():
    """
    Test KS test for insertion indices with empty input array
    """
    out = utils.compute_indices_ks_test([], 1000, mode='D+')
    assert all(o is None for o in out)


@pytest.mark.parametrize("ndims, radius", [(2, 1), (10, 2), (10, 10), (1, 1)])
def test_draw_surface_nsphere(ndims, radius):
    """
    Assert the correct number of samples with the correct dimensions are drawn
    and all lie on the surface.
    """
    out = utils.draw_surface_nsphere(ndims, r=radius, N=1000)

    assert out.shape[0] == 1000
    assert out.shape[1] == ndims
    np.testing.assert_allclose(np.sqrt(np.sum(out ** 2., axis=1)), radius)


@pytest.mark.parametrize("ndims, radius", [(2, 1), (10, 2), (10, 10), (1, 1)])
def test_draw_nball(ndims, radius):
    """
    Assert the correct number of samples with the correct dimensions are drawn
    and all lie within the n-ball.
    """
    out = utils.draw_nsphere(ndims, r=radius, N=1000)

    assert out.shape[0] == 1000
    assert out.shape[1] == ndims
    np.testing.assert_array_less(np.sqrt(np.sum(out ** 2, axis=-1)), radius)


def test_get_uniform_distribution_cpu():
    """
    Test function for getting uniform torch distrbution over n dimensions
    when called on cpu
    """
    dist = utils.get_uniform_distribution(10, 1, 'cpu')
    assert dist.sample().get_device() == -1


@cuda
def test_get_uniform_distribution_cuda():
    """
    Test function for getting uniform torch distrbution over n dimensions
    when called on CUDA
    """
    dist = utils.get_uniform_distribution(10, 1, device='cuda')
    assert dist.sample().get_device() != -1


@pytest.mark.parametrize("r, var, fuzz", [
    (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 2.0, 1.0), (4.0, 2.0, 1.5),
    (7.0, 4.0, 2.0)])
@pytest.mark.flaky(run=5)
def test_draw_truncated_gaussian_1d(r, var, fuzz):
    """
    Test drawing from a truncated Gaussian in 1d
    """
    s = utils.draw_truncated_gaussian(1, r, var=var, N=1000, fuzz=fuzz)
    sigma = np.sqrt(var)
    d = stats.truncnorm(-r * fuzz / sigma, r * fuzz / sigma,
                        loc=0, scale=sigma)
    _, p = stats.kstest(np.squeeze(s), d.cdf)
    assert p >= 0.05
