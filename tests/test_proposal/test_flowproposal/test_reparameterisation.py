# -*- coding: utf-8 -*-
"""Test methods related to reparameterisations"""
import numpy as np
from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import FlowProposal
from nessai.reparameterisations import (
    get_reparameterisation
)
import pytest
from unittest.mock import MagicMock, patch


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


@patch('nessai.reparameterisations.CombinedReparameterisation')
def test_configure_reparameterisations_dict(mocked_class, proposal):
    """Test configuration for reparameterisations dictionary"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model = MagicMock
    proposal.model.bounds = {'x': [-1, 1], 'y': [-1, 1]}
    proposal.names = ['x', 'y']
    FlowProposal.configure_reparameterisations(
        proposal, {'x': {'reparameterisation': 'default'}})

    proposal.add_default_reparameterisations.assert_called_once()
    assert proposal.rescaled_names == ['x_prime', 'y']
    assert proposal.rescale_parameters == ['x']
    assert proposal._reparameterisation.parameters == ['x', 'y']
    assert proposal._reparameterisation.prime_parameters == ['x_prime', 'y']
    assert mocked_class.called_once


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
    x = numpy_array_to_live_points(np.random.randn(n, 2), ['x', 'y'])
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
