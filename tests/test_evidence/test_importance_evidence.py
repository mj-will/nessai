# -*- coding: utf-8 -*-
"""
Tests related to the evidence computation in the importance sampler
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec

from nessai.evidence import _INSIntegralState as State


@pytest.fixture
def state():
    return create_autospec(State)


@pytest.mark.parametrize('normalised', [False, True])
def test_init(state, normalised):
    """Assert the initial value are correct"""
    State.__init__(state, normalised=normalised)
    assert state.normalised is normalised
    assert state._logZ == np.NINF
    assert state._n == 0
    # Must be None for logic to work.
    assert state._log_meta_constant is None


def test_update_evidence(state):
    """Assert the evidence is correctly updated"""
    x = np.array([(1, 2), (3, 4)], dtype=[('logL', 'f8'), ('logW', 'f8')])
    state._logZ_history = np.array([5, 6])
    state._n = 2
    logZ_history = np.array([5, 6, 3, 7])

    State.update_evidence(state, x)

    # Should agree with logsumexp of logZ history
    assert state._logZ == 7.419716646315339
    assert state._n == 4
    np.testing.assert_array_equal(
        state._logZ_history,
        logZ_history,
    )


def test_update_evidence_from_nested_samples(state):
    """Assert the evidence is correctly updated from nested samples"""
    x = np.array([(1, 2), (3, 4)], dtype=[('logL', 'f8'), ('logW', 'f8')])
    logZ_history = np.array([3, 7])

    State.update_evidence_from_nested_samples(state, x)

    # Should agree with logsumexp of logZ history
    assert state._logZ == 7.0181499279178094
    assert state._n == 2
    np.testing.assert_array_equal(
        state._logZ_history,
        logZ_history,
    )


@pytest.mark.parametrize('constant', [0.0, 0.5])
def test_renormalise_true(state, constant):
    """Assert that renormalise returns True when the constant is set"""
    state._log_meta_constant = constant
    state.normalised = False
    assert State.renormalise.__get__(state) is True


def test_renormalise_false(state):
    """Assert that renormalise returns False when the consant is not set
    and normalised is False.
    """
    state._log_meta_constant = None
    state.normalised = True
    State.renormalise.__get__(state) is False


def test_renormalise_error(state):
    """Assert an error is raised if the log-meta proposal is not set."""
    state.normalised = False
    state._log_meta_constant = None
    with pytest.raises(RuntimeError) as excinfo:
        State.renormalise.__get__(state)
    assert 'constant is not set' in str(excinfo.value)


def test_log_meta_constant_normalised(state):
    """Assert that log(n) is returned if the samples are normalised"""
    state._log_meta_constant = None
    state.normalised = True
    state._n = 10
    assert State.log_meta_constant.__get__(state) == np.log(10)


def test_log_meta_constant_meta_proposal(state):
    """Assert the log meta proposal is returned if set"""
    state._log_meta_constant = 0.5
    assert State.log_meta_constant.__get__(state) == 0.5


def test_log_meta_constant_not_normalised(state):
    """
    Assert an error is raised if the weights are not normalised but the meta
    proposal constant isn't set.
    """
    state._log_meta_constant = None
    state.normalised = False
    with pytest.raises(RuntimeError) as excinfo:
        State.log_meta_constant.__get__(state)
    assert 'Samples are not correctly normalised' in str(excinfo.value)


def test_log_meta_constant_setter_not_normalised(state):
    """Assert the constant can be set if normalised is False"""
    state.normalised = False
    State.log_meta_constant.__set__(state, 0.5)
    assert state._log_meta_constant == 0.5


def test_log_meta_constant_setter_normalised(state):
    """Assert the constant cannot be set if normalised is True"""
    state.normalised = True
    with pytest.raises(RuntimeError) as excinfo:
        State.log_meta_constant.__set__(state, 0.5)
    assert 'Cannot set the meta constant' in str(excinfo.value)


def test_log_constant_not_renormalise(state):
    """Assert log-constant is zero if the evidence is already normalised"""
    state.renormalise = False
    assert State.log_constant.__get__(state) == 0.0


def test_log_constant_renormalise(state):
    """Assert log-constant is the sum of meta constant and log(n)"""
    state.renormalise = True
    state._n = 10
    state._log_meta_constant = 0.5
    assert State.log_constant.__get__(state) == (0.5 - np.log(10))


def test_logZ(state):
    """
    Assert the log-evidence is the sum of _logZ and the normalisation
    constant.
    """
    state._logZ = -5.0
    state.log_constant = -4.0
    assert State.logZ.__get__(state) == -9.0


def test_log_evidence():
    """Assert log-evidence method is just an alias"""
    assert State.log_evidence is State.logZ


def test_log_evidence_error():
    """Assert the log-evidence errors calls `compute_uncertainty`"""
    state.compute_uncertainty = MagicMock(return_value=0.1)
    assert State.log_evidence_error.__get__(state) == 0.1
    state.compute_uncertainty.assert_called_once()


def test_compute_update_log_Z_no_renormalise(state):
    """
    Assert the update log-evidence is computed correctly in the default case.

    Should just logsumexp the two evidences
    """
    x = np.array([(1, 2), (3, 4)], dtype=[('logL', 'f8'), ('logW', 'f8')])
    state._logZ = 6.313261687518223     # logsumexp([5, 6])
    state.renormalise = False
    out = State.compute_updated_log_Z(state, x)
    assert out == 7.419716646315339


def test_compute_update_log_Z_renormalise(state):
    """
    Assert the update log-evidence is computed correctly when renormalising
    """
    x = np.array([(1, 2), (3, 4)], dtype=[('logL', 'f8'), ('logW', 'f8')])
    state._logZ = 6.313261687518223     # logsumexp([5, 6])
    state.renormalise = True
    state._log_meta_constant = np.log(7)
    state._n = 2
    out = State.compute_updated_log_Z(state, x)
    expected = (7.419716646315339 + (np.log(7) - np.log(4)))
    assert out == expected


def test_compute_condition(state):
    """Assert the condition is computed correctly

    Assert `compute_updated_log_Z` is called with correct inputs
    """
    samples = np.array([1, 2])
    state.compute_updated_log_Z = MagicMock(return_value=0.5)
    state.logZ = 0.1

    assert State.compute_condition(state, samples) == 0.4

    state.compute_updated_log_Z.assert_called_once_with(samples)


@pytest.mark.parametrize('samples', [None, np.array([])])
def test_compute_condition_no_samples(state, samples):
    """Assert condition is zero if no samples are provided"""
    assert State.compute_condition(state, samples) == 0.0


def test_compute_uncertainity(state):
    """Assert uncertainty calculation gives the expected value"""
    state._logZ_history = np.array([1, 2, 3, 4])
    state.log_meta_constant = np.log(4)
    state._n = 4
    state.logZ = 4.440189698561196

    Z_hat = np.exp(state.logZ)
    Z = np.exp(state._logZ_history + np.log(4))

    # Should sqrt(sum((Z - Z_hat) ** 2) / n * (n - 1)) / Z_hat
    expected = float(np.sqrt(
        np.sum((Z - Z_hat) ** 2.0)
        / 12
    ) / Z_hat)

    err = State.compute_uncertainty(state)
    assert err > 0.0
    np.testing.assert_equal(err, expected)


def test_log_posterior_weights(state):
    """Assert the log-posterior weights are correct."""
    val = 4.440189698561196
    state._logZ_history = np.array([1, 2, 3, 4])
    state.logZ = val + 2.0 - np.log(4)
    state.log_meta_constant = 2.0
    log_w = State.log_posterior_weights.__get__(state)
    np.testing.assert_array_almost_equal(
        log_w,
        np.array([1, 2, 3, 4]) - val + np.log(4),
        decimal=10,
    )


@pytest.mark.parametrize(
    'log_w', [np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0])]
)
def test_ess(state, log_w):
    """Assert ess is computed correctly"""
    state.log_posterior_weights = log_w
    ess = State.effective_n_posterior_samples.__get__(state)
    assert ess == 2.086110772843278
