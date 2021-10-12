# -*- coding: utf-8 -*-
"""
Test the base proposal class.
"""
import datetime
import logging
import pickle
import numpy as np
import pytest
from unittest.mock import Mock, call, create_autospec, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal.base import (
    Proposal,
    _initialize_global_variables,
    _log_likelihood_wrapper
)


class DummyProposal(Proposal):

    def draw(self, old_param):
        pass


@pytest.fixture
def proposal():
    return create_autospec(Proposal)


def test_init(proposal):
    """Test the init method"""
    model = Mock()
    Proposal.__init__(proposal, model, n_pool=2)
    assert proposal.model is model
    assert proposal.training_count == 0
    assert proposal._checked_population is True
    assert proposal.population_time.total_seconds() == 0
    assert proposal.logl_eval_time.total_seconds() == 0


def test_init_with_draw():
    """Assert an error is raised if `draw` is not implemented."""
    model = Mock()
    with pytest.raises(TypeError) as excinfo:
        Proposal(model)
    assert "class Proposal with abstract method" in str(excinfo.value)


def test_initialised(proposal):
    """Test the initialised property"""
    proposal._initialised = True
    val = Proposal.initialised.__get__(proposal)
    assert val is True


@pytest.mark.parametrize('val', [True, False])
def test_initialised_setter(proposal, val):
    """Test the setter for initialised."""
    Proposal.initialised.__set__(proposal, val)
    assert proposal._initialised is val


def test_initialise(proposal):
    """Test the initialise method"""
    Proposal.initialise(proposal)
    assert proposal.initialised is True


def test_configure_pool(proposal):
    """Test configuring the pool"""
    model = Mock()
    proposal.model = model
    proposal.n_pool = 4
    proposal.check_acceptance = False
    proposal.pool = None
    pool = Mock()
    with patch('multiprocessing.Pool', return_value=pool) as mock_pool:
        Proposal.configure_pool(proposal)
    proposal.check_acceptance is True
    assert proposal.pool is pool
    mock_pool.assert_called_once_with(
        processes=4,
        initializer=_initialize_global_variables,
        initargs=(model,)
    )


def test_configure_pool_none(proposal, caplog):
    """Test configuring the pool when n_pool is None"""
    caplog.set_level(logging.INFO)
    proposal.n_pool = None
    proposal.pool = None
    Proposal.configure_pool(proposal)
    assert proposal.pool is None
    assert 'n_pool is none, no multiprocessing pool' in str(caplog.text)


@pytest.mark.parametrize('code', [10, 2])
def test_close_pool(proposal, code):
    """Test closing the pool"""
    pool = Mock()
    pool.close = Mock()
    pool.terminate = Mock()
    pool.join = Mock()
    proposal.pool = pool
    Proposal.close_pool(proposal, code=code)
    pool.join.assert_called_once()
    if code == 2:
        pool.terminate.assert_called_once()
        pool.pool.assert_not_called()
    else:
        pool.close.assert_called_once()
        pool.terminate.assert_not_called()
    assert proposal.pool is None


def test_evaluate_likelihoods_pool(proposal):
    """Test evaluating the likelihood with a pool"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = np.array([3, 4])
    proposal.pool = Mock(side_effect=True)
    proposal.pool.map = Mock(return_value=logL)
    proposal.samples = samples
    proposal.logl_eval_time = datetime.timedelta()
    proposal.model = Mock()
    proposal.model.likelihood_evaluations = 100
    Proposal.evaluate_likelihoods(proposal)
    proposal.pool.map.assert_called_once_with(
        _log_likelihood_wrapper,
        samples
    )
    proposal.logl_eval_time.total_seconds() > 0
    assert proposal.model.likelihood_evaluations == 102
    np.testing.assert_array_equal(proposal.samples['logL'], logL)


def test_evaluate_likelihoods_no_pool(proposal):
    """Test evaluating the likelihood without a pool"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ['x'])
    logL = np.array([3, 4])
    proposal.pool = None
    proposal.samples = samples
    proposal.logl_eval_time = datetime.timedelta()
    proposal.model = Mock()
    proposal.model.evaluate_log_likelihood = Mock(side_effect=logL)
    Proposal.evaluate_likelihoods(proposal)
    proposal.model.evaluate_log_likelihood.assert_has_calls(
        [call(samples[0]), call(samples[1])]
    )
    proposal.logl_eval_time.total_seconds() > 0
    np.testing.assert_array_equal(proposal.samples['logL'], logL)


def test_draw(proposal):
    """Assert an error is raised."""
    with pytest.raises(NotImplementedError):
        Proposal.draw(proposal, None)


def test_test_draw(proposal):
    """Test the test draw method"""
    proposal.model = Mock()
    proposal.model.new_point = Mock(return_value=1)
    proposal.model.log_prior = Mock(return_value=-1)

    new_point = numpy_array_to_live_points(np.array([[1]]), ['x'])
    new_point['logP'] = -1
    proposal.draw = Mock(return_value=new_point)
    Proposal.test_draw(proposal)

    proposal.model.new_point.assert_called_once()
    proposal.draw.assert_called_once_with(1)
    proposal.model.log_prior.assert_called_once_with(new_point)


def test_test_draw_error(proposal):
    """Test the test draw method with an incorrect prior value"""
    proposal.model = Mock()
    proposal.model.new_point = Mock(return_value=1)
    proposal.model.log_prior = Mock(return_value=-1)

    new_point = numpy_array_to_live_points(np.array([[1]]), ['x'])
    new_point['logP'] = -2
    proposal.draw = Mock(return_value=new_point)
    with pytest.raises(RuntimeError) as excinfo:
        Proposal.test_draw(proposal)
    assert 'Log prior of new point is incorrect!' in str(excinfo.value)

    proposal.model.new_point.assert_called_once()
    proposal.draw.assert_called_once_with(1)
    proposal.model.log_prior.assert_called_once_with(new_point)


def test_train(proposal, caplog):
    """Test the train method."""
    caplog.set_level(logging.INFO)
    Proposal.train(proposal, 1, plot=True)
    assert 'This proposal method cannot be trained' in str(caplog.text)


def test_resume(proposal):
    """Test the resume method."""
    model = Mock()
    Proposal.resume(proposal, model)
    assert proposal.model is model


def test_getstate(proposal):
    """Test the get state method called by pickle."""
    proposal.model = Mock()
    proposal.pool = Mock()
    state = Proposal.__getstate__(proposal)
    assert state['pool'] is None
    assert 'model' not in list(state.keys())


@pytest.mark.integration_test
def test_pickling(model):
    """Test pickling the proposal"""
    proposal = DummyProposal(model, n_pool=1)
    proposal.configure_pool()
    d = pickle.dumps(proposal)
    out = pickle.loads(d)
    proposal.close_pool()
    assert out.pool is None
    assert hasattr(out, 'model') is False
