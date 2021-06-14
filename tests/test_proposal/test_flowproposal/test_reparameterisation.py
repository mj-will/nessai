# -*- coding: utf-8 -*-
"""Test methods related to reparameterisations"""
import numpy as np
from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import FlowProposal
from nessai.reparameterisations import (
    get_reparameterisation
)
import pytest
from unittest.mock import MagicMock, call, patch


@pytest.fixture
def dummy_rc():
    """Dummy reparameteristation class"""
    m = MagicMock()
    m.__name__ = 'DummyReparameterisation'
    return m


@pytest.fixture
def dummy_cmb_rc():
    """Dummy combined reparameteristation class"""
    m = MagicMock()
    m.add_reparameterisation = MagicMock()
    return m


def test_default_reparameterisation(proposal):
    """Test to make sure default reparameterisation does not cause errors
    for default proposal.
    """
    FlowProposal.add_default_reparameterisations(proposal)


@patch('nessai.reparameterisations.get_reparameterisation')
def test_get_reparamaterisation(mocked_fn, proposal):
    """Make sure the underlying function is called"""
    FlowProposal.get_reparameterisation(proposal, 'angle')
    assert mocked_fn.called_once_with('angle')


def test_configure_reparameterisations_dict(proposal, dummy_cmb_rc, dummy_rc):
    """Test configuration for reparameterisations dictionary.

    Also tests to make sure boundary inversion is set.
    """
    dummy_rc.return_value = 'r'
    # Need to add the parameters before hand to prevent a
    # NullReparameterisation from being added
    dummy_cmb_rc.parameters = ['x']
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(return_value=(
        dummy_rc, {'boundary_inversion': True}
    ))
    proposal.model = MagicMock()
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x']

    with patch('nessai.proposal.flowproposal.CombinedReparameterisation',
               return_value=dummy_cmb_rc) as mocked_class:
        FlowProposal.configure_reparameterisations(
            proposal, {'x': {'reparameterisation': 'default'}})

    proposal.get_reparameterisation.assert_called_once_with('default')
    proposal.add_default_reparameterisations.assert_called_once()
    dummy_rc.assert_called_once_with(
        prior_bounds={'x': [-1, 1]},
        parameters='x',
        boundary_inversion=True
    )
    mocked_class.assert_called_once()
    proposal._reparameterisation.add_reparameterisations.\
        assert_called_once_with('r')

    assert proposal.boundary_inversion is True
    assert proposal.names == ['x']


@patch('nessai.proposal.flowproposal.CombinedReparameterisation')
def test_configure_reparameterisations_dict_w_params(mocked_class, proposal,
                                                     dummy_rc, dummy_cmb_rc):
    """Test configuration for reparameterisations dictionary with parameters.

    For example:

        {'x': {'reparmeterisation': 'default', 'parameters': 'y'}}

    This should add both x and y to the reparameterisation.
    """
    dummy_rc.return_value = 'r'
    # Need to add the parameters before hand to prevent a
    # NullReparameterisation from being added
    dummy_cmb_rc.parameters = ['x', 'y']
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(return_value=(
        dummy_rc, {},
    ))
    proposal.model = MagicMock()
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']

    with patch('nessai.proposal.flowproposal.CombinedReparameterisation',
               return_value=dummy_cmb_rc) as mocked_class:
        FlowProposal.configure_reparameterisations(
            proposal,
            {'x': {'reparameterisation': 'default', 'parameters': ['y']}}
        )

    proposal.get_reparameterisation.assert_called_once_with('default')
    proposal.add_default_reparameterisations.assert_called_once()
    dummy_rc.assert_called_once_with(
        prior_bounds={'x': [-1, 1], 'y': [-1, 1]},
        parameters=['y', 'x'],
    )
    mocked_class.assert_called_once()
    proposal._reparameterisation.add_reparameterisations.\
        assert_called_once_with('r')

    assert proposal.names == ['x', 'y']


@patch('nessai.reparameterisations.CombinedReparameterisation')
def test_configure_reparameterisations_dict_missing(mocked_class, proposal):
    """
    Test configuration for reparameterisations dictionary when missing
    the reparameterisation for a parameter.

    This should raise a runtime error.
    """
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model = MagicMock
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(
            proposal, {'x': {'scale': 1.0}})

    assert 'No reparameterisation found for x' in str(excinfo.value)


@patch('nessai.reparameterisations.CombinedReparameterisation')
def test_configure_reparameterisations_str(mocked_class, proposal):
    """Test configuration for reparameterisations dictionary from a str"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model = MagicMock
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']
    FlowProposal.configure_reparameterisations(
        proposal, {'x': 'default'})

    proposal.add_default_reparameterisations.assert_called_once()
    assert proposal.rescaled_names == ['x_prime', 'y']
    assert proposal.rescale_parameters == ['x']
    assert proposal._reparameterisation.parameters == ['x', 'y']
    assert proposal._reparameterisation.prime_parameters == ['x_prime', 'y']
    assert mocked_class.called_once


@patch('nessai.reparameterisations.CombinedReparameterisation')
def test_configure_reparameterisations_dict_reparam(mocked_class, proposal):
    """Test configuration for reparameterisations dictionary"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model = MagicMock
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']
    FlowProposal.configure_reparameterisations(
        proposal, {'default': {'parameters': ['x']}})

    proposal.add_default_reparameterisations.assert_called_once()
    assert proposal.rescaled_names == ['x_prime', 'y']
    assert proposal.rescale_parameters == ['x']
    assert proposal._reparameterisation.parameters == ['x', 'y']
    assert proposal._reparameterisation.prime_parameters == ['x_prime', 'y']
    assert mocked_class.called_once


def test_configure_reparameterisations_incorrect_type(proposal):
    """Assert an error is raised when input is not a dictionary"""
    with pytest.raises(TypeError) as excinfo:
        FlowProposal.configure_reparameterisations(proposal, ['default'])
    assert 'must be a dictionary' in str(excinfo.value)


def test_configure_reparameterisations_incorrect_config_type(proposal):
    """Assert an error is raised when the config for a key is not a dictionary
    or a known reparameterisation.
    """
    proposal.names = ['x']
    with pytest.raises(TypeError) as excinfo:
        FlowProposal.configure_reparameterisations(proposal, {'x': ['a']})
    assert 'Unknown config type' in str(excinfo.value)


@pytest.mark.parametrize(
    'reparam',
    [{'z': {'reparameterisation': 'sine'}}, {'sine': {'parameters': ['x']}}]
)
def test_configure_reparameterisation_unknown(proposal, reparam):
    """
    Assert an error is raised if an unknown reparameterisation or parameters
    is passed.
    """
    proposal.names = ['x']
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(proposal, reparam)
    assert 'is not a parameter in the model or a known' in str(excinfo.value)


def test_configure_reparameterisation_no_parameters(proposal, dummy_rc):
    """Assert an error is raised if no parameters are specified"""
    proposal.names = ['x']
    proposal.get_reparameterisation = MagicMock(return_value=(
        dummy_rc, {},
    ))
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(
            proposal, {'default': {'update_bounds': True}})
    assert 'No parameters key' in str(excinfo.value)


def test_set_rescaling_with_model(proposal, model):
    """
    Test setting the rescaling when the model contains reparmaeterisations.
    """
    proposal.model = model
    proposal.model.reparameterisations = {'x': 'default'}
    proposal.set_boundary_inversion = MagicMock()

    def update(self):
        proposal.rescale_parameters = ['x']
        proposal.rescaled_names = ['x_prime']

    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    FlowProposal.set_rescaling(proposal)

    proposal.set_boundary_inversion.assert_called_once()
    proposal.configure_reparameterisations.assert_called_once_with(
        {'x': 'default'}
    )
    assert proposal.reparameterisations == {'x': 'default'}
    assert proposal.rescaled_names == ['x_prime']


def test_set_rescaling_with_reparameterisations(proposal, model):
    """
    Test setting the rescaling when a reparameterisations dict is defined.
    """
    proposal.model = model
    proposal.model.reparameterisations = None
    proposal.reparameterisations = {'x': 'default'}
    proposal.set_boundary_inversion = MagicMock()

    def update(self):
        proposal.rescale_parameters = ['x']
        proposal.rescaled_names = ['x_prime']

    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    FlowProposal.set_rescaling(proposal)

    proposal.set_boundary_inversion.assert_called_once()
    proposal.configure_reparameterisations.assert_called_once_with(
        {'x': 'default'}
    )
    assert proposal.reparameterisations == {'x': 'default'}
    assert proposal.rescaled_names == ['x_prime']


@pytest.mark.parametrize('update_bounds', [True, False])
@pytest.mark.parametrize(
    'rescale_parameters',
    [
        (False, ['x', 'y']),
        (True, ['x_prime', 'y_prime']),
        (['x'], ['x_prime', 'y'])
    ]
)
def test_set_rescaling_parameters_list(proposal, model, update_bounds,
                                       rescale_parameters):
    """Test setting rescaling without reparameterisations"""
    proposal.model = model
    proposal.model.reparameterisations = None
    proposal.reparameterisations = None
    proposal.configure_reparameterisations = MagicMock()

    proposal.rescale_parameters = rescale_parameters[0]
    proposal.rescale_bounds = [-1, 1]
    proposal.update_bounds = update_bounds

    FlowProposal.set_rescaling(proposal)

    proposal.configure_reparameterisations.assert_not_called()
    assert proposal.rescaled_names == rescale_parameters[1]

    if rescale_parameters[0]:
        assert proposal._rescale_factor == 2.0
        assert proposal._rescale_shift == -1.0
        assert proposal._min == {'x': -5, 'y': -5}
        assert proposal._max == {'x': 5, 'y': 5}


@pytest.mark.parametrize('n', [1, 10])
def test_rescale_w_reparameterisation(proposal, n):
    """Test rescaling when using reparameterisation dict"""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y'])
    x['logL'] = np.random.randn(n)
    x['logP'] = np.random.randn(n)
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2), ['x_prime', 'y_prime'])
    proposal.x_prime_dtype = \
        [('x_prime', 'f8'), ('y_prime', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.reparameterise = MagicMock(return_value=[
        x, x_prime, np.ones(x.size)])

    x_prime_out, log_j = \
        FlowProposal._rescale_w_reparameterisation(
            proposal, x, compute_radius=False, test='lower')

    np.testing.assert_array_equal(
        x_prime[['x_prime', 'y_prime']], x_prime_out[['x_prime', 'y_prime']])
    np.testing.assert_array_equal(
        x[['logP', 'logL']], x_prime_out[['logL', 'logP']])
    proposal._reparameterisation.reparameterise.assert_called_once()


@pytest.mark.parametrize('n', [1, 10])
def test_inverse_rescale_w_reparameterisation(proposal, n):
    """Test rescaling when using reparameterisation dict"""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y']).squeeze()
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2), ['x_prime', 'y_prime'])
    x_prime['logL'] = np.random.randn(n)
    x_prime['logP'] = np.random.randn(n)
    proposal.x_dtype = \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.inverse_reparameterise = \
        MagicMock(return_value=[x, x_prime, np.ones(x.size)])

    x_out, log_j = \
        FlowProposal._inverse_rescale_w_reparameterisation(
            proposal, x_prime)

    np.testing.assert_array_equal(x[['x', 'y']], x_out[['x', 'y']])
    np.testing.assert_array_equal(
        x_prime[['logP', 'logL']], x_out[['logL', 'logP']])
    proposal._reparameterisation.inverse_reparameterise.assert_called_once()


@pytest.mark.parametrize('n', [1, 10])
@pytest.mark.parametrize('compute_radius', [False, True])
def test_rescale_to_bounds(proposal, model, n, compute_radius):
    """Test the default rescaling to bounds.

    Also tests the log Jacobian determinant.
    """
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y']).squeeze()
    x_prime_expected = numpy_array_to_live_points(
        np.zeros([n, 2]), ['x_prime', 'y_prime'])

    x_prime_expected['x_prime'] = 2 * (x['x'] + 5) / 10 - 1
    x_prime_expected['y_prime'] = 2 * (x['y'] + 4) / 8 - 1

    proposal.x_prime_dtype = \
        [('x_prime', 'f8'), ('y_prime', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime']
    proposal.model = model
    proposal.boundary_inversion = []

    proposal._rescale_factor = 2.0
    proposal._rescale_shift = -1.0
    proposal._min = {'x': -5, 'y': -4}
    proposal._max = {'x': 5, 'y': 4}

    x_prime, log_j = \
        FlowProposal._rescale_to_bounds(
            proposal, x, compute_radius=compute_radius)

    np.testing.assert_equal(log_j, -np.log(20))
    np.testing.assert_array_equal(x_prime, x_prime_expected)


@pytest.mark.parametrize('n', [1, 10])
def test_inverse_rescale_to_bounds(proposal, model, n):
    """Test the default method for the inverse rescaling.

    Also tests the log Jacobian determinant.
    """
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2),
        ['x_prime', 'y_prime']
    ).squeeze()
    x_expected = \
        numpy_array_to_live_points(np.zeros([n, 2]), ['x', 'y'])

    x_expected['x'] = 10.0 * (x_prime['x_prime'] + 1.0) / 2.0 - 5.0
    x_expected['y'] = 8.0 * (x_prime['y_prime'] + 1.0) / 2.0 - 4.0

    proposal.x_dtype = \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime']
    proposal.model = model
    proposal.boundary_inversion = []

    proposal._rescale_factor = 2.0
    proposal._rescale_shift = -1.0
    proposal._min = {'x': -5, 'y': -4}
    proposal._max = {'x': 5, 'y': 4}

    x, log_j = \
        FlowProposal._inverse_rescale_to_bounds(proposal, x_prime)

    np.testing.assert_equal(log_j, np.log(20))
    np.testing.assert_array_equal(x, x_expected)


@pytest.mark.parametrize('has_inversion', [False, True])
def test_verify_rescaling(proposal, has_inversion):
    """Test the method that tests the rescaling at runtime"""
    x = np.array([[1], [2]], dtype=[('x', 'f8')])
    x_prime = x['x'] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()
    log_j_inv = np.array([2, 2])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])

    proposal.model = MagicMock()
    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))
    proposal.check_state = MagicMock()

    FlowProposal.verify_rescaling(proposal)

    proposal.check_state.assert_has_calls(4 * [call(x)])
    # Should call 4 different test cases
    calls = [
        call(x, test='lower'),
        call(x, test='upper'),
        call(x, test=False),
        call(x, test=None)
    ]
    proposal.rescale.assert_has_calls(calls)
    proposal.inverse_rescale.assert_has_calls(4 * [call(x_prime)])


@pytest.mark.parametrize('has_inversion', [False, True])
def test_verify_rescaling_invertible_error(proposal, has_inversion):
    """Assert an error is raised if the rescaling is not invertible"""
    x = np.array([[1], [2]], dtype=[('x', 'f8')])
    x_prime = x['x'] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()[::-1]
    log_j_inv = np.array([2, 2])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])

    proposal.model = MagicMock()
    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.verify_rescaling(proposal)
    assert 'Rescaling is not invertible for x' in str(excinfo.value)


def test_verify_rescaling_duplicate_error(proposal):
    """Assert an error is raised if the duplication is missing samples"""
    x = np.array([[1], [2]], dtype=[('x', 'f8')])
    x_prime = x['x'] / 2
    log_j = np.array([-2, -2, -2, -2])
    x_out = np.array([[1], [3], [4], [5]], dtype=[('x', 'f8')])
    log_j_inv = np.array([2, 2, 2, 2])

    proposal.model = MagicMock()
    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.verify_rescaling(proposal)
    assert 'Duplicate samples must map to same input' in str(excinfo.value)


@pytest.mark.parametrize('has_inversion', [False, True])
def test_verify_rescaling_jacobian_error(proposal, has_inversion):
    """Assert an error is raised if the Jacobian is not invertible"""
    x = np.array([[1], [2]], dtype=[('x', 'f8')])
    x_prime = x['x'] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()
    log_j_inv = np.array([2, 1])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])

    proposal.model = MagicMock()
    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.verify_rescaling(proposal)
    assert 'Rescaling Jacobian is not invertible' in str(excinfo.value)
