# -*- coding: utf-8 -*-
"""
Test functions related to drawing posterior samples
"""
import logging
import numpy as np
import pytest
from unittest.mock import patch

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


@pytest.mark.parametrize("nlive", [10, 10 * np.ones(20)])
@pytest.mark.parametrize("expectation", ["logt", "t"])
def test_compute_weights(nlive, expectation):
    """Test computing the weights for set of likelihood values."""
    log_l = np.random.randn(20)

    log_z, log_w = compute_weights(log_l, nlive, expectation=expectation)

    assert len(log_w) == len(log_l)
    assert np.isfinite(log_z)


@pytest.mark.parametrize("expectation", ["logt", "t"])
def test_compute_weights_correct_weights(expectation):
    """Assert the weights (X_i) are correct"""
    nlive = 10
    log_l = np.random.randn(20)
    out = -np.log1p(1 / nlive) * np.ones(len(log_l))
    with patch("numpy.log1p", return_value=out) as mock_log1p, patch(
        "nessai.posterior.logsubexp", return_value=np.random.rand(21)
    ), patch("nessai.posterior.log_integrate_log_trap", return_value=0.0):
        compute_weights(log_l, nlive=nlive, expectation=expectation)
    if expectation == "t":
        mock_log1p.assert_called_once()
    else:
        mock_log1p.assert_not_called()


def test_compute_weights_invalid_nlive():
    """Assert an error is raised if nlive does not match the logs-likelihood"""
    with pytest.raises(
        ValueError, match=r"nlive and samples are different lengths"
    ):
        compute_weights([1, 2, 3], [4, 5])


def test_compute_weights_invalid_expectation():
    """Assert an error is raised if expectation is invalid"""
    with pytest.raises(
        ValueError, match=r"Expectation must be t or logt, got: a"
    ):
        compute_weights(np.random.randn(10), 10, expectation="a")


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


@pytest.mark.slow_integration_test
def test_compute_weights_vs_log_posterior_weights(model, tmp_path):
    """Test the two different methods for computing posterior weights and
    assert they return the sames value.

    Checks both the values of the weights and log-evidence.
    """
    from nessai.flowsampler import FlowSampler

    output = tmp_path / "posterior_comparison"
    output.mkdir()
    fs = FlowSampler(
        model, output=output, nlive=100, checkpointing=False, plot=False
    )
    fs.run(save=False, plot=False)

    log_z_0, log_w_0 = compute_weights(fs.nested_samples["logL"], fs.ns.nlive)
    log_w_1 = fs.ns.state.log_posterior_weights

    np.testing.assert_array_equal(log_w_1, log_w_0)
    assert log_z_0 == fs.ns.state.log_evidence
