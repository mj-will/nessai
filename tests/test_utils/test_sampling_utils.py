# -*- coding: utf-8 -*-
"""
Test utilities for sampling in the latent space.
"""
import numpy as np
import pytest
from scipy import stats, special
from unittest.mock import create_autospec, patch

from nessai.utils.sampling import (
    NDimensionalTruncatedGaussian,
    compute_radius,
    draw_gaussian,
    draw_nsphere,
    draw_surface_nsphere,
    draw_truncated_gaussian,
    draw_uniform,
)


def test_compute_radius():
    """Assert compute radius calls the correct function"""
    with patch("scipy.stats.chi.ppf") as mock:
        compute_radius(2, 0.95)
    mock.assert_called_once_with(0.95, 2)


@pytest.mark.parametrize(
    "d, q, r",
    [
        [1, 0.6827, 1.0],
        [1, 0.9545, 2.0],
        [2, 0.3935, 1.0],
        [5, 0.8909, 3.0],
        [10, 0.9004, 4.0],
    ],
)
def test_compute_radius_integration(d, q, r):
    """Assert compute radius returns the expected value.

    Checks that for a given confidence level the correct radius is returned,
    values checked correspond to standard deviations.

    Values for q taken from: \
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4358977/
    """
    r_out = compute_radius(d, q)
    np.testing.assert_almost_equal(r_out, r, decimal=4)


@pytest.mark.parametrize("ndims, radius", [(2, 1), (10, 2), (10, 10), (1, 1)])
def test_draw_surface_nsphere(ndims, radius):
    """
    Assert the correct number of samples with the correct dimensions are drawn
    and all lie on the surface.
    """
    out = draw_surface_nsphere(ndims, r=radius, N=1000)

    assert out.shape[0] == 1000
    assert out.shape[1] == ndims
    np.testing.assert_allclose(np.sqrt(np.sum(out**2.0, axis=1)), radius)


@pytest.mark.parametrize("ndims, radius", [(2, 1), (10, 2), (10, 10), (1, 1)])
def test_draw_nball(ndims, radius):
    """
    Assert the correct number of samples with the correct dimensions are drawn
    and all lie within the n-ball.
    """
    out = draw_nsphere(ndims, r=radius, N=1000)

    assert out.shape[0] == 1000
    assert out.shape[1] == ndims
    np.testing.assert_array_less(np.sqrt(np.sum(out**2, axis=-1)), radius)


def test_draw_uniform():
    """Assert the underlying numpy function is called correctly."""
    expected = np.array([0.5, 1])
    with patch("numpy.random.uniform", return_value=expected) as mock:
        out = draw_uniform(2, r=1, N=100, fuzz=2)
    mock.assert_called_once_with(0, 1, (100, 2))
    np.testing.assert_array_equal(out, expected)


def test_draw_gaussian():
    """Assert the underlying numpy function is called correctly."""
    expected = np.array([1, 2])
    with patch("numpy.random.randn", return_value=expected) as mock:
        out = draw_gaussian(2, r=1, N=100, fuzz=2)
    mock.assert_called_once_with(100, 2)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "r, var, fuzz",
    [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (2.0, 2.0, 1.0),
        (4.0, 2.0, 1.5),
        (7.0, 4.0, 2.0),
    ],
)
@pytest.mark.flaky(reruns=5)
def test_draw_truncated_gaussian_1d(r, var, fuzz):
    """
    Test drawing from a truncated Gaussian in 1d
    """
    s = draw_truncated_gaussian(1, r, var=var, N=2000, fuzz=fuzz)
    sigma = np.sqrt(var)
    d = stats.truncnorm(
        -r * fuzz / sigma, r * fuzz / sigma, loc=0, scale=sigma
    )
    _, p = stats.kstest(np.squeeze(s), d.cdf)
    assert p >= 0.05


@pytest.mark.parametrize("dims", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("radius", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("fuzz", [1.0, 1.1, 1.5])
def test_ndimensional_truncated_gaussian_u_max(dims, radius, fuzz):
    """Check the value of the u_max"""
    dist = create_autospec(NDimensionalTruncatedGaussian)

    expected_u_max = special.gammainc(dims / 2, (radius * fuzz) ** 2 / 2)

    NDimensionalTruncatedGaussian.__init__(dist, dims, radius, fuzz)

    assert dist.u_max == expected_u_max


@pytest.mark.parametrize(
    "r, fuzz",
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 1.0),
        (4.0, 1.5),
        (7.0, 2.0),
    ],
)
@pytest.mark.flaky(reruns=5)
def test_ndimensional_truncated_gaussian_sample(r, fuzz):
    """Assert samples are correctly drawn from the distribution"""
    dist = NDimensionalTruncatedGaussian(1, r, fuzz=fuzz)
    s = dist.sample(10_000)
    d = stats.truncnorm(-r * fuzz, r * fuzz, loc=0)
    _, p = stats.kstest(np.squeeze(s), d.cdf)
    assert p >= 0.05
