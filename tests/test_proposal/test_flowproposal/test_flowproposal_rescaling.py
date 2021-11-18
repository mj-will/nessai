# -*- coding: utf-8 -*-
"""Test methods related to reparameterisations"""
import numpy as np
from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import FlowProposal
from nessai.reparameterisations import (
    NullReparameterisation,
    get_reparameterisation,
)
import pytest
from unittest.mock import MagicMock, Mock, call, patch


@pytest.fixture
def proposal(proposal):
    """Specific mocked proposal for reparameterisation tests"""
    proposal.use_default_reparameterisations = False
    return proposal


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


def test_configure_reparameterisations_requires_prime_prior(
        proposal,
        dummy_rc,
        dummy_cmb_rc
):
    """
    Test configuration that requires a prime prior but the prime prior is
    missing.
    """
    dummy_rc.return_value = 'r'
    # Need to add the parameters before hand to prevent a
    # NullReparameterisation from being added
    dummy_cmb_rc.parameters = ['x', 'y']
    dummy_cmb_rc.has_prime_prior = False
    dummy_cmb_rc.requires_prime_prior = True
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(return_value=(
        dummy_rc, {},
    ))
    proposal.model = MagicMock()
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']

    with patch('nessai.proposal.flowproposal.CombinedReparameterisation',
               return_value=dummy_cmb_rc), \
         pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(
            proposal,
            {'x': {'reparameterisation': 'default', 'parameters': ['y']}}
        )

    assert 'One or more reparameterisations require ' in str(excinfo.value)


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


@patch('nessai.reparameterisations.CombinedReparameterisation')
def test_configure_reparameterisations_none(mocked_class, proposal):
    """Test configuration when input is None"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model = MagicMock()
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']
    FlowProposal.configure_reparameterisations(proposal, None)
    proposal.add_default_reparameterisations.assert_called_once()
    assert proposal.rescaled_names == ['x', 'y']

    assert proposal.rescale_parameters == []
    assert proposal._reparameterisation.parameters == ['x', 'y']
    assert proposal._reparameterisation.prime_parameters == ['x', 'y']
    assert all(
        [isinstance(r, NullReparameterisation)
         for r in proposal._reparameterisation.reparameterisations.values()]
    )
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
    """Test setting rescaling without reparameterisations."""
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


def test_default_rescale(proposal):
    """Test the default rescaling method."""
    x = np.array([[1, 2], [3, 4]])
    x_prime, log_j = FlowProposal.rescale(proposal, x)
    np.testing.assert_array_equal(x, x_prime)
    assert (log_j == 0.).all()


def test_default_inverse_rescale(proposal):
    """Test the default inverse rescaling method."""
    x_prime = np.array([[1, 2], [3, 4]])
    x, log_j = FlowProposal.inverse_rescale(proposal, x_prime)
    np.testing.assert_array_equal(x, x_prime)
    assert (log_j == 0.).all()


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
    model.names.append('z')
    model.bounds['z'] = [-10, 10]
    x = numpy_array_to_live_points(
        np.random.randn(n, 3), ['x', 'y', 'z']
    ).squeeze()
    x_prime_expected = numpy_array_to_live_points(
        np.zeros([n, 3]), ['x_prime', 'y_prime', 'z'])

    x_prime_expected['x_prime'] = 2 * (x['x'] + 5) / 10 - 1
    x_prime_expected['y_prime'] = 2 * (x['y'] + 4) / 8 - 1
    x_prime_expected['z'] = x['z']

    proposal.x_prime_dtype = \
        [('x_prime', 'f8'), ('y_prime', 'f8'), ('z', 'f8'), ('logP', 'f8'),
         ('logL', 'f8')]

    proposal.names = ['x', 'y', 'z']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime', 'z']
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
@pytest.mark.parametrize('inputs', [('split', True), ('duplicate', False)])
@pytest.mark.parametrize('bound', ['lower', 'upper'])
def test_rescale_to_bounds_w_inversion_duplicate(
    proposal,
    model,
    n,
    inputs,
    bound
):
    """Test the default rescaling with inversion using duplication."""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y']).squeeze()
    x_prime_expected = numpy_array_to_live_points(
        np.zeros([n, 2]), ['x_prime', 'y_prime'])

    x_prime_expected['x_prime'] = (x['x'] + 5) / 10
    x_prime_expected['y_prime'] = (x['y'] + 4) / 8

    if bound == 'lower':
        x_prime_expected = np.concatenate([x_prime_expected, x_prime_expected])
        x_prime_expected['x_prime'][n:] *= -1
    elif bound == 'upper':
        x_prime_expected['x_prime'] = 1 - x_prime_expected['x_prime']
        x_prime_expected = np.concatenate([x_prime_expected, x_prime_expected])
        x_prime_expected['x_prime'][n:] *= -1

    proposal.x_prime_dtype = \
        [('x_prime', 'f8'), ('y_prime', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime']
    proposal.model = model
    proposal.boundary_inversion = ['x']

    proposal._rescale_factor = 1.0
    proposal._rescale_shift = 0.0
    proposal._min = {'x': -5, 'y': -4}
    proposal._max = {'x': 5, 'y': 4}
    proposal._edges = {'x': None}
    proposal.detect_edges_kwargs = {'k': 2}
    proposal.inversion_type = inputs[0]

    with patch('nessai.proposal.flowproposal.detect_edge',
               return_value=bound) as mock_detect_edge:
        x_prime, log_j = \
            FlowProposal._rescale_to_bounds(
                proposal, x, compute_radius=inputs[1], test=True)

    np.testing.assert_array_equal(
        mock_detect_edge.call_args[0][0],
        x_prime_expected['x_prime'][:n]
    )
    assert mock_detect_edge.call_args[1]['test'] is True
    assert mock_detect_edge.call_args[1]['k'] == 2

    np.testing.assert_equal(log_j, -np.log(80))
    np.testing.assert_array_equal(x_prime, x_prime_expected)


@pytest.mark.parametrize('n', [1, 10])
@pytest.mark.parametrize('bound', ['lower', 'upper'])
def test_rescale_to_bounds_w_inversion_split(
    proposal,
    model,
    n,
    bound
):
    """Test the default rescaling with inversion using splitting."""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y']).squeeze()
    x_prime_expected = numpy_array_to_live_points(
        np.zeros([n, 2]), ['x_prime', 'y_prime'])

    x_prime_expected['x_prime'] = (x['x'] + 5) / 10
    x_prime_expected['y_prime'] = (x['y'] + 4) / 8

    inv = np.random.choice(n, n // 2, replace=False)

    if bound == 'lower':
        x_prime_expected['x_prime'][inv] *= -1
    elif bound == 'upper':
        x_prime_expected['x_prime'] = 1 - x_prime_expected['x_prime']
        x_prime_expected['x_prime'][inv] *= -1

    proposal.x_prime_dtype = \
        [('x_prime', 'f8'), ('y_prime', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime']
    proposal.model = model
    proposal.boundary_inversion = ['x']

    proposal._rescale_factor = 1.0
    proposal._rescale_shift = 0.0
    proposal._min = {'x': -5, 'y': -4}
    proposal._max = {'x': 5, 'y': 4}
    proposal._edges = {'x': None}
    proposal.detect_edges_kwargs = {'k': 2}
    proposal.inversion_type = 'split'

    with patch('nessai.proposal.flowproposal.detect_edge',
               return_value=bound) as mock_detect_edge, \
         patch('numpy.random.choice', return_value=inv):
        x_prime, log_j = \
            FlowProposal._rescale_to_bounds(
                proposal, x, compute_radius=False, test=True)

    np.testing.assert_array_equal(
        mock_detect_edge.call_args[0][0],
        x_prime_expected['x_prime'][:n]
    )
    assert mock_detect_edge.call_args[1]['test'] is True
    assert mock_detect_edge.call_args[1]['k'] == 2

    np.testing.assert_equal(log_j, -np.log(80))
    np.testing.assert_array_equal(x_prime, x_prime_expected)


@pytest.mark.parametrize('n', [1, 10])
@pytest.mark.parametrize('bound', ['lower', 'upper'])
def test_rescale_to_bounds_w_inversion_false(
    proposal,
    model,
    n,
    bound
):
    """
    Test the default rescaling without inversion and with edges already set.
    """
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y']).squeeze()
    x_prime_expected = numpy_array_to_live_points(
        np.zeros([n, 2]), ['x_prime', 'y_prime'])

    x_prime_expected['x_prime'] = (x['x'] + 5) / 10
    x_prime_expected['y_prime'] = (x['y'] + 4) / 8

    proposal.x_prime_dtype = \
        [('x_prime', 'f8'), ('y_prime', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime']
    proposal.model = model
    proposal.boundary_inversion = ['x']

    proposal._rescale_factor = 1.0
    proposal._rescale_shift = 0.0
    proposal._min = {'x': -5, 'y': -4}
    proposal._max = {'x': 5, 'y': 4}
    proposal._edges = {'x': False}
    proposal.detect_edges_kwargs = {'k': 2}
    proposal.inversion_type = 'split'

    with patch('nessai.proposal.flowproposal.detect_edge',
               return_value=bound) as mock_detect_edge:
        x_prime, log_j = \
            FlowProposal._rescale_to_bounds(
                proposal, x, compute_radius=False, test=True)

    mock_detect_edge.assert_not_called()

    np.testing.assert_equal(log_j, -np.log(80))
    np.testing.assert_array_equal(x_prime, x_prime_expected)


@pytest.mark.parametrize('n', [1, 10])
def test_inverse_rescale_to_bounds(proposal, model, n):
    """Test the default method for the inverse rescaling.

    Also tests the log Jacobian determinant.
    """
    model.names.append('z')
    model.bounds['z'] = [-10, 10]
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 3),
        ['x_prime', 'y_prime', 'z']
    ).squeeze()
    x_expected = \
        numpy_array_to_live_points(np.zeros([n, 3]), ['x', 'y', 'z'])

    x_expected['x'] = 10.0 * (x_prime['x_prime'] + 1.0) / 2.0 - 5.0
    x_expected['y'] = 8.0 * (x_prime['y_prime'] + 1.0) / 2.0 - 4.0
    x_expected['z'] = x_prime['z']

    proposal.x_dtype = \
        [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y', 'z']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime', 'z']
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


@pytest.mark.parametrize('n', [1, 10])
@pytest.mark.parametrize('itype', ['lower', 'upper'])
def test_inverse_rescale_to_bounds_w_inversion(proposal, model, n, itype):
    """Test the default method for the inverse rescaling with inversion.

    Also tests the log Jacobian determinant.
    """
    model.names.append('z')
    model.bounds['z'] = [-10, 10]
    x_prime = numpy_array_to_live_points(
        np.random.uniform(-1, 1, (n, 3)),
        ['x_prime', 'y_prime', 'z']
    ).squeeze()

    x_expected = \
        numpy_array_to_live_points(np.zeros([n, 3]), ['x', 'y', 'z'])

    if itype == 'lower':
        x_expected['x'] = 10.0 * np.abs(x_prime['x_prime']) - 5.0
    elif itype == 'upper':
        x_expected['x'] = 10.0 * (1 - np.abs(x_prime['x_prime'])) - 5.0

    x_expected['y'] = 8.0 * x_prime['y_prime'] - 4.0
    x_expected['z'] = x_prime['z']

    proposal.x_dtype = \
        [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('logP', 'f8'), ('logL', 'f8')]

    proposal.names = ['x', 'y', 'z']
    proposal.rescale_parameters = ['x', 'y']
    proposal.rescaled_names = ['x_prime', 'y_prime', 'z']
    proposal.model = model
    proposal.boundary_inversion = ['x']
    proposal.inversion_type = 'split'
    proposal._edges = {'x': itype}

    proposal._rescale_factor = 1.0
    proposal._rescale_shift = 0.0
    proposal._min = {'x': -5, 'y': -4}
    proposal._max = {'x': 5, 'y': 4}

    x, log_j = \
        FlowProposal._inverse_rescale_to_bounds(proposal, x_prime)

    np.testing.assert_equal(log_j, np.log(80))
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


def test_check_state_boundary_inversion_default(proposal):
    """Test the check state method for boundary inversion"""
    x = numpy_array_to_live_points(np.random.randn(10, 2), ['x', 'y'])
    proposal._reparameterisation = None
    proposal._edges = {'x': 'lower', 'y': 'upper'}
    proposal.boundary_inversion = ['x', 'y']
    proposal.update_bounds = False
    FlowProposal.check_state(proposal, x)
    assert all(v is None for v in proposal._edges.values())


def test_check_state_boundary_inversion_reparameterisations(proposal):
    """
    Test the check state method for boundary inversion with
    reparameterisations.
    """
    x = numpy_array_to_live_points(np.random.randn(10, 2), ['x', 'y'])
    proposal._reparameterisation = Mock()
    proposal._reparameterisation.reset_inversion = MagicMock()
    proposal.boundary_inversion = ['x', 'y']
    proposal.update_bounds = False
    FlowProposal.check_state(proposal, x)
    proposal._reparameterisation.reset_inversion.assert_called_once()


def test_check_state_update_bounds_default(proposal, model):
    """
    Test the check state method for updating bounds.
    """
    x = numpy_array_to_live_points(np.random.randn(10, 2), ['x', 'y'])
    proposal.model = model
    excepted_min = (np.min(x['x']), np.min(x['y']))
    excepted_max = (np.max(x['x']), np.max(x['y']))
    proposal._reparameterisation = None
    proposal.boundary_inversion = False
    proposal.update_bounds = True
    proposal._min = {'x': -np.inf, 'y': -np.inf}
    proposal._max = {'x': np.inf, 'y': np.inf}
    FlowProposal.check_state(proposal, x)

    assert proposal._min['x'] == excepted_min[0]
    assert proposal._min['y'] == excepted_min[1]
    assert proposal._max['x'] == excepted_max[0]
    assert proposal._max['y'] == excepted_max[1]


def test_check_state_update_bounds_reparameterisations(proposal):
    """
    Test the check state method for updating bounds.
    """
    x = numpy_array_to_live_points(np.random.randn(10, 2), ['x', 'y'])
    proposal._reparameterisation = Mock()
    proposal._reparameterisation.update_bounds = MagicMock()
    proposal.boundary_inversion = False
    proposal.update_bounds = True

    FlowProposal.check_state(proposal, x)
    proposal._reparameterisation.update_bounds.assert_called_once()
