# -*- coding: utf-8 -*-
"""Test the GW flow proposal method"""
import pytest
from unittest.mock import create_autospec, MagicMock, patch

from nessai.gw.proposal import GWFlowProposal, AugmentedGWFlowProposal


@pytest.fixture
def proposal():
    return create_autospec(GWFlowProposal)


@pytest.fixture
def augmented_proposal():
    return create_autospec(AugmentedGWFlowProposal)


def test_get_reparameterisation(proposal):
    """Test to make sure the correct version of get reparameterisation is
    called.
    """
    with patch('nessai.gw.proposal.get_gw_reparameterisation',
               return_value='reparam') as mock:
        out = GWFlowProposal.get_reparameterisation(proposal, 'default')
    assert out == 'reparam'
    mock.assert_called_once_with('default')


def test_add_default_reparameterisation(proposal):
    """Test the method `add_default_reparameterisation`"""
    proposal.aliases = GWFlowProposal.aliases
    proposal.names = ['chirp_mass', 'theta_jn']
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ['chirp_mass']
    proposal.model = MagicMock()
    proposal.model.bounds = \
        {'chirp_mass': [10.0, 20.0], 'theta_jn': [0.0, 3.0]}

    reparam = MagicMock()
    with patch('nessai.gw.proposal.get_gw_reparameterisation',
               return_value=(reparam, {})) as mock_get:
        GWFlowProposal.add_default_reparameterisations(proposal)

    mock_get.assert_called_once_with('angle-sine')
    reparam.assert_called_once_with(parameters=['theta_jn'],
                                    prior_bounds={'theta_jn': [0.0, 3.0]})


def test_augmented_get_reparameterisation(augmented_proposal):
    """Test to make sure the correct version of get reparameterisation is
    called.
    """
    with patch('nessai.gw.proposal.get_gw_reparameterisation',
               return_value='reparam') as mock:
        out = AugmentedGWFlowProposal.get_reparameterisation(
            augmented_proposal, 'default')
    assert out == 'reparam'
    mock.assert_called_once_with('default')


def test_augmented_reparameterisation_prior(augmented_proposal):
    """Test to make sure the correct components of the log prior are being
    called.

    These are:
    - agumented_prior
    - FlowProposal.log_prior
    """
    augmented_proposal.augmented_prior = MagicMock()
    augmented_proposal._reparameterisation = MagicMock()
    augmented_proposal.model = MagicMock()

    with patch('nessai.proposal.flowproposal.FlowProposal.log_prior') as mock:
        AugmentedGWFlowProposal.log_prior(augmented_proposal, 1)

    augmented_proposal.augmented_prior.assert_called_once_with(1)
    mock.assert_called_once_with(1)


def test_augmented_reparameterisation_prime_prior(augmented_proposal):
    """Test to make sure the correct components of the primed_log prior are
    being called.

    These are:
    - agumented_prior
    - FlowProposal.x_prime_log_prior
    """
    augmented_proposal.augmented_prior = MagicMock()
    augmented_proposal._reparameterisation = MagicMock()

    with patch('nessai.proposal.flowproposal.FlowProposal.'
               'x_prime_log_prior') as mock:
        AugmentedGWFlowProposal.x_prime_log_prior(augmented_proposal, 1)

    augmented_proposal.augmented_prior.assert_called_once_with(1)
    mock.assert_called_once_with(1)
