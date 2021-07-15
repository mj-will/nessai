# -*- coding: utf-8 -*-
"""
Test utilities for sampling in the latent space.
"""
import numpy as np
import pytest
from scipy import stats

from nessai.utils.sampling import (
    draw_nsphere,
    draw_surface_nsphere,
    draw_truncated_gaussian,
)


@pytest.mark.parametrize("ndims, radius", [(2, 1), (10, 2), (10, 10), (1, 1)])
def test_draw_surface_nsphere(ndims, radius):
    """
    Assert the correct number of samples with the correct dimensions are drawn
    and all lie on the surface.
    """
    out = draw_surface_nsphere(ndims, r=radius, N=1000)

    assert out.shape[0] == 1000
    assert out.shape[1] == ndims
    np.testing.assert_allclose(np.sqrt(np.sum(out ** 2., axis=1)), radius)


@pytest.mark.parametrize("ndims, radius", [(2, 1), (10, 2), (10, 10), (1, 1)])
def test_draw_nball(ndims, radius):
    """
    Assert the correct number of samples with the correct dimensions are drawn
    and all lie within the n-ball.
    """
    out = draw_nsphere(ndims, r=radius, N=1000)

    assert out.shape[0] == 1000
    assert out.shape[1] == ndims
    np.testing.assert_array_less(np.sqrt(np.sum(out ** 2, axis=-1)), radius)


@pytest.mark.parametrize(
    "r, var, fuzz",
    [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (2.0, 2.0, 1.0),
        (4.0, 2.0, 1.5),
        (7.0, 4.0, 2.0)
    ]
)
@pytest.mark.flaky(run=5)
def test_draw_truncated_gaussian_1d(r, var, fuzz):
    """
    Test drawing from a truncated Gaussian in 1d
    """
    s = draw_truncated_gaussian(1, r, var=var, N=1000, fuzz=fuzz)
    sigma = np.sqrt(var)
    d = stats.truncnorm(-r * fuzz / sigma, r * fuzz / sigma,
                        loc=0, scale=sigma)
    _, p = stats.kstest(np.squeeze(s), d.cdf)
    assert p >= 0.05
