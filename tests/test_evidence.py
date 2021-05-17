# -*- coding: utf-8 -*-
"""
Test the object that handles the nested sampling evidence and prior volumes.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec, MagicMock

from nessai.evidence import (
    _NSIntegralState,
    logsubexp,
    LogNegativeError
)


@pytest.fixture()
def nlive():
    return 100


@pytest.fixture
def point():
    return {'logL': -10.0, 'logW': 0.}


@pytest.fixture
def second_point():
    return {'logL': -5.0, 'logW': 0.}


@pytest.fixture
def state():
    return create_autospec(_NSIntegralState)


def test_logsubexp():
    """Test the values returned by logsubexp"""
    out = logsubexp(2, 1)
    np.testing.assert_almost_equal(out, np.log(np.exp(2) - np.exp(1)),
                                   decimal=12)


def test_logsubexp_negative():
    """
    Test behaviour of logsubexp for x < y
    """
    with pytest.raises(LogNegativeError):
        logsubexp(1, 2)


def test_log_weights(state):
    """Test the log-weights method"""
    state.logLs = [-np.inf, -10, -5]
    state.log_vols = [0.0, -0.2, -0.4]
    log_w = _NSIntegralState.log_weights(state)
    assert len(log_w) == 2


def test_importance_weights(state):
    """Test the importance weights"""
    state.logLs = [-np.inf, -10, -5]
    state.log_vols = [0.0, -0.2, -0.4]
    state.log_w_norm = [0.1, 0.1]
    state.logZ = -0.5
    state.log_weights = \
        MagicMock(return_value=_NSIntegralState.log_weights(state))
    w, zw, pw = _NSIntegralState.importance_weights(state, G=0.9)


def test_increment(point, nlive):
    """Test the basic functionality of incrementing the evidence estimate"""
    state = _NSIntegralState(nlive)
    state.increment(point)

    assert np.isclose(state.logX, float(-1 / nlive))
    assert state.logZ != -np.inf
    np.testing.assert_equal(state.logLs, [-np.inf, -10])


def test_increment_w_weights(point, nlive):
    """Test incrementing the evidence estimate when using weights"""
    state = _NSIntegralState(nlive)
    point['logW'] = -5.
    state.increment(point, log_w_norm=-0.5)

    assert np.isclose(state.logX, -np.exp(-4.5))
    assert state.logZ != -np.inf
    np.testing.assert_equal(state.logLs, [-np.inf, -10])


def test_increment_weights_error(point, nlive):
    """
    Test to make sure an error is raised if the weights are non-zero when
    the normalisation has not been given (rejection sampling).
    """
    state = _NSIntegralState(nlive)
    point['logW'] = -5.
    with pytest.raises(ValueError) as excinfo:
        state.increment(point)

    assert 'Weights must be zero' in str(excinfo.value)


def test_finalise(point, nlive):
    """Test the check that finalise returns an improved logZ"""
    state = _NSIntegralState(nlive)
    state.increment(point)
    pre = state.logZ
    state.finalise()
    assert state.logZ != -np.inf
    assert pre != state.logZ


def test_info(point, second_point, nlive):
    """Test to check the information increases as expected"""
    state = _NSIntegralState(nlive)
    state.increment(point)
    assert state.info == [0., 0.]
    state.increment(second_point)
    assert state.info[2] > 0


def test_track_gradients(point, second_point, nlive):
    """
    Test to make sure gradients are not computed when tracking is disabled
    """
    state = _NSIntegralState(nlive, track_gradients=False)
    state.increment(point)
    state.increment(second_point)
    assert len(state.gradients) == 1


def test_variable_nlive(point, nlive):
    """
    Test to make sure that the using a different nlive changes the update
    values.
    """
    state = _NSIntegralState(nlive)
    state.increment(point, nlive=50)
    assert np.isclose(state.logX, float(-1 / 50))


def test_plot(point, second_point, nlive):
    """Test the plotting function"""
    state = _NSIntegralState(nlive)
    state.increment(point)
    state.increment(second_point)
    fig = state.plot()
    assert fig is not None


def test_plot_w_filename(point, second_point, nlive, tmpdir):
    """Test the plotting function with a filename specified"""
    filename = str(tmpdir.mkdir('test'))
    state = _NSIntegralState(nlive)
    state.increment(point)
    state.increment(second_point)
    fig = state.plot(filename=filename)
    assert fig is None
