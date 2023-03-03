# -*- coding: utf-8 -*-
"""
Test the object that handles the nested sampling evidence and prior volumes.
"""
from unittest.mock import create_autospec, patch
import numpy as np
import pytest

from nessai.evidence import (
    _BaseNSIntegralState,
    _NSIntegralState,
    logsubexp,
)


@pytest.fixture()
def nlive():
    return 100


@pytest.fixture()
def base_state():
    return create_autospec(_BaseNSIntegralState)


@pytest.fixture()
def ns_state():
    return create_autospec(_NSIntegralState)


def test_logsubexp_negative():
    """
    Test behaviour of logsubexp for x < y
    """
    with pytest.raises(Exception):
        logsubexp(1, 2)


@pytest.mark.parametrize(
    "method", ["log_evidence", "log_evidence_error", "log_posterior_weights"]
)
def test_base_state_not_implemented_error_properties(base_state, method):
    """Assert all of the abstract properties raised not implemented errors"""
    with pytest.raises(NotImplementedError):
        getattr(_BaseNSIntegralState, method).__get__(base_state)


@pytest.mark.parametrize(
    "weights, expected", [([], 0), ([0.1, 0.2, 0.3, 0.4], (1 / 0.3))]
)
def test_base_state_effective_n_posterior_samples(
    base_state, weights, expected
):
    """Assert the correct effective sample size is returned.

    Should be zero if weights are empty.
    """
    base_state.log_posterior_weights = np.log(weights)
    ess = _BaseNSIntegralState.effective_n_posterior_samples.__get__(
        base_state
    )
    np.testing.assert_almost_equal(ess, expected, decimal=10)


def test_invalid_expectation(ns_state):
    """Assert an error is raised if `expectation` is an invalid value"""
    with pytest.raises(
        ValueError, match=r"Expectation must be t or logt, got: a"
    ):
        _NSIntegralState.__init__(ns_state, 100, expectation="a")


@pytest.mark.parametrize("expectation", ["logt", "t"])
def test_increment(nlive, expectation):
    """Test the basic functionality of incrementing the evidence estimate"""
    state = _NSIntegralState(nlive, expectation=expectation)
    state.increment(-10)

    if expectation == "logt":
        target = -1 / nlive
    else:
        target = -np.log1p(1 / nlive)

    assert state.logw == target
    assert state.logZ != -np.inf
    np.testing.assert_equal(state.logLs, [-np.inf, -10])


def test_increment_monotonic_warning(ns_state, caplog):
    """Assert a warning is raised if the likelihood is non-monotonic"""
    ns_state.logLs = [1, 2, 3]
    ns_state.base_nlive = 10
    ns_state.nlive = 3 * [10]
    ns_state.logZ = -2.0
    ns_state.logw = 1.0
    ns_state.info = [0]
    ns_state.log_vols = []
    ns_state.track_gradients = False
    ns_state.expectation = "logt"
    _NSIntegralState.increment(ns_state, 2.5)
    assert "received non-monotonic logL" in str(caplog.text)


def test_log_evidence(ns_state):
    """Assert the log-evidence property returns the correct value"""
    expected = 1.0
    ns_state.logZ = expected
    out = _NSIntegralState.log_evidence.__get__(ns_state)
    assert out == expected


def test_log_evidence_error(ns_state, nlive):
    """Assert the log-evidence error property returns the correct value"""
    expected = np.sqrt(10 / nlive)
    ns_state.info = [1, 5, 10]
    ns_state.base_nlive = nlive
    out = _NSIntegralState.log_evidence_error.__get__(ns_state)
    assert out == expected


def test_finalise(nlive):
    """Test the check that finalise returns an improved logZ"""
    state = _NSIntegralState(nlive)
    state.increment(-10)
    pre = state.logZ
    state.finalise()
    assert state.logZ != -np.inf
    assert pre != state.logZ


def test_info(nlive):
    """Test to check the information increases as expected"""
    state = _NSIntegralState(nlive)
    state.increment(-10)
    assert state.info == [0.0]
    state.increment(-5)
    assert state.info[1] > 0


def test_track_gradients(nlive):
    """
    Test to make sure gradients are not computed when tracking is disabled
    """
    state = _NSIntegralState(nlive, track_gradients=False)
    state.increment(-10)
    state.increment(-5)
    assert len(state.gradients) == 1


@pytest.mark.parametrize(
    "expectation, value", [("logt", -1 / 50), ("t", -np.log1p(1 / 50))]
)
def test_variable_nlive(nlive, expectation, value):
    """
    Test to make sure that the using a different nlive changes the update
    values.
    """
    state = _NSIntegralState(nlive, expectation=expectation)
    state.increment(-10, nlive=50)
    assert state.logw == value


def test_plot(nlive):
    """Test the plotting function"""
    state = _NSIntegralState(nlive)
    state.increment(-10)
    state.increment(-5)
    fig = state.plot()
    assert fig is not None


def test_plot_w_filename(nlive, tmpdir):
    """Test the plotting function with a filename specified"""
    filename = str(tmpdir.mkdir("test"))
    state = _NSIntegralState(nlive)
    state.increment(-10)
    state.increment(-5)
    fig = state.plot(filename=filename)
    assert fig is None


def test_log_posterior_weights(ns_state):
    """Test the log-posterior weights property"""
    log_vols = [0.0, -0.1, -0.2, -0.3]
    logL = [np.NINF, -10.0, -5.0, -0.0]
    ns_state.log_vols = log_vols
    ns_state.logLs = logL
    log_z = -1.0
    with patch(
        "nessai.evidence.log_integrate_log_trap", return_value=log_z
    ) as mock_int:
        out = _NSIntegralState.log_posterior_weights.__get__(ns_state)

    trap_inputs = mock_int.call_args[0]
    np.testing.assert_array_equal(trap_inputs[0], logL + [-0.0])
    np.testing.assert_array_equal(trap_inputs[1], log_vols + [np.NINF])
    # Output should one shorted than logL since the initial point is not a
    # nested sample.
    assert len(out) == (len(logL) - 1)
