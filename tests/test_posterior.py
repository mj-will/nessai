# -*- coding: utf-8 -*-
"""
Test functions related to drawing posterior samples
"""
import logging
import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.posterior import compute_weights, draw_posterior_samples
from nessai.utils.testing import assert_structured_arrays_equal

resampling_options = [
    "rejection_sampling",
    "importance_sampling",
    "multinomial_resampling",
]


@pytest.fixture()
def ns():
    return numpy_array_to_live_points(np.random.randn(20, 1), ["x"])


@pytest.fixture(params=resampling_options)
def method(request):
    return request.param


def test_compute_weights():
    """Test computing the weights for set of likelihood values"""
    log_l = np.random.randn(20)

    log_z, log_w = compute_weights(log_l, 10)

    assert len(log_w) == len(log_l)
    assert np.isfinite(log_z)


def test_draw_posterior_samples(ns, method):
    """Test drawing posterior samples."""
    ns["logL"] = np.log(np.random.rand(ns.size))
    ns["logP"] = np.zeros(ns.size)
    p = draw_posterior_samples(ns, nlive=10, method=method)
    assert np.isin(p, ns).all()


def test_draw_posterior_samples_w_weights(ns, method):
    """Test drawing samples when weights are specified"""
    log_w = np.log(np.random.rand(len(ns)))
    p = draw_posterior_samples(ns, log_w=log_w, method=method)
    assert len(p) > 0


def test_draw_posterior_samples_with_n(caplog, ns, method):
    """Test drawing samples with n specified"""
    n = 10
    log_w = np.log(np.random.rand(len(ns)))
    with caplog.at_level(logging.WARNING):
        post = draw_posterior_samples(ns, log_w=log_w, method=method, n=n)
    if method == "rejection_sampling":
        assert "Number of samples cannot be specified" in caplog.text
    else:
        assert len(post) == n


def test_draw_posterior_samples_indices(ns, method):
    """Assert the indices are returned when return_indices=True"""
    log_w = np.log(np.random.rand(len(ns)))
    post, indices = draw_posterior_samples(
        ns,
        log_w=log_w,
        method=method,
        return_indices=True,
    )
    assert_structured_arrays_equal(post, ns[indices])


def test_draw_posterior_unknown_method(ns):
    """Assert an error is raised if the method is not known"""
    with pytest.raises(ValueError) as excinfo:
        draw_posterior_samples(ns, nlive=10, method="not_a_method")
    assert "Unknown method of drawing posterior samples: not_a_method" in str(
        excinfo.value
    )
