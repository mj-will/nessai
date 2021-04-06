# -*- coding: utf-8 -*-
"""Test methods related to reparameterisations"""
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
