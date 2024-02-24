"""Test the INS evidence"""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
from scipy.special import logsumexp

from nessai.evidence import (
    _INSIntegralState as INSState,
    log_evidence_from_ins_samples,
)


@pytest.fixture
def state():
    return create_autospec(INSState)


@pytest.fixture
def live_points(model, ins_parameters):
    x = model.new_point(2)
    x["logP"] = model.log_prior(x)
    x["logL"] = model.log_likelihood(x)
    x["logQ"] = model.new_point_log_prob(x)
    x["logW"] = x["logP"] - x["logQ"]
    return x


@pytest.fixture
def nested_samples(model, ins_parameters):
    x = model.new_point(4)
    x["logP"] = model.log_prior(x)
    x["logL"] = model.log_likelihood(x)
    x["logQ"] = model.new_point_log_prob(x)
    x["logW"] = x["logP"] - x["logQ"]
    return x


def test_init(state):
    """Test the init method"""
    INSState.__init__(state)
    assert state._logZ == np.NINF


def test_log_Z(state):
    """Assert the normalisation is applied"""
    state._logZ = -10.0
    state._n = 100
    assert INSState.logZ.__get__(state) == (-10.0 - np.log(100))


def test_log_evidence_error(state):
    """Assert property calls the the correct method"""
    state.compute_uncertainty = MagicMock(return_value=0.1)
    assert INSState.log_evidence_error.__get__(state) == 0.1
    state.compute_uncertainty.assert_called_once_with(log_evidence=True)


def test_log_evidence_live_points(state):
    """Check the correct value is returned"""
    state._weights_lp = np.log(np.array([1, 2, 3]))
    np.testing.assert_almost_equal(
        INSState.log_evidence_live_points.__get__(state), np.log(2)
    )


def test_log_evidence_live_points_error(state):
    """Assert an error is raised if the live points are None"""
    state._weights_lp = None
    with pytest.raises(RuntimeError, match="Live points are not set"):
        INSState.log_evidence_live_points.__get__(state)


def test_log_evidence_live_nested_samples(state):
    """Check the correct value is returned"""
    state._weights_ns = np.log(np.array([1, 2, 3]))
    np.testing.assert_almost_equal(
        INSState.log_evidence_nested_samples.__get__(state), np.log(2)
    )


def test_log_posterior_weights(state):
    """Check the posterior weights are correct"""
    state._weights = np.array([1, 2, 3])
    state.logZ = -1.0
    out = INSState.log_posterior_weights.__get__(state)
    np.testing.assert_equal(out, np.array([2, 3, 4]))


def test_update_evidence_ns_obly(state, nested_samples):
    """Test updating the evidence"""
    x = nested_samples
    expected = logsumexp(x["logL"] + x["logW"])
    assert np.isfinite(expected)
    INSState.update_evidence(state, nested_samples, live_points=None)
    np.testing.assert_equal(state._logZ, expected)
    assert state._n == x.size
    assert state._weights_lp is None


def test_update_evidence_both(state, nested_samples, live_points):
    """Test updating the evidence"""
    x = np.concatenate([nested_samples, live_points])
    expected = logsumexp(x["logL"] + x["logW"])
    print(x)
    assert np.isfinite(expected)
    INSState.update_evidence(state, nested_samples, live_points=live_points)
    np.testing.assert_equal(state._logZ, expected)
    assert state._n == x.size
    assert state._weights_lp is not None


@pytest.mark.parametrize("ns_only, expected", [(True, 1.0), (False, 2.0)])
def test_compute_evidence_ratio(state, ns_only, expected):
    """Assert the correct value is returned"""
    state.log_evidence_live_points = -1.0
    state.log_evidence_nested_samples = -2.0
    state.logZ = -3.0
    assert INSState.compute_evidence_ratio(state, ns_only) == expected


def test_compute_uncertainty_log(state):
    """Assert a finite value is returned"""
    state._weights = -np.array([1, 2, 3])
    state._n = 3
    state.logZ = -6.0
    out_ln = INSState.compute_uncertainty(state, log_evidence=True)
    out = INSState.compute_uncertainty(state, log_evidence=False)
    # Check errors are equivalent
    np.testing.assert_almost_equal(out_ln, out / np.exp(state.logZ))


def test_log_evidence_ins_samples():
    n = 10
    log_l = np.log(np.random.rand(n))
    log_w = np.log(np.random.rand(n))
    samples = np.array(
        [*zip(log_l, log_w)], dtype=[("logL", "f8"), ("logW", "f8")]
    )

    expected = np.log(np.mean(np.exp(log_l + log_w)))

    out = log_evidence_from_ins_samples(samples)

    np.testing.assert_almost_equal(out, expected, decimal=12)
