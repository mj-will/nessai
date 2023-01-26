# -*- coding: utf-8 -*-
"""Test the GW flow proposal method"""
from math import pi
import numpy as np
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
    with patch(
        "nessai.gw.proposal.get_gw_reparameterisation", return_value="reparam"
    ) as mock:
        out = GWFlowProposal.get_reparameterisation(proposal, "default")
    assert out == "reparam"
    mock.assert_called_once_with("default")


def test_add_default_reparameterisation(proposal):
    """Test the method `add_default_reparameterisation`"""
    proposal.aliases = GWFlowProposal.aliases
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["chirp_mass"]
    proposal.model = MagicMock()
    proposal.model.names = ["chirp_mass", "theta_jn"]
    proposal.model.bounds = {
        "chirp_mass": [10.0, 20.0],
        "theta_jn": [0.0, 3.0],
    }

    reparam = MagicMock()
    reparam.__name__ = "MockReparam"
    with patch(
        "nessai.gw.proposal.get_gw_reparameterisation",
        return_value=(reparam, {}),
    ) as mock_get:
        GWFlowProposal.add_default_reparameterisations(proposal)

    mock_get.assert_called_once_with("angle-sine")
    reparam.assert_called_once_with(
        parameters=["theta_jn"], prior_bounds={"theta_jn": [0.0, 3.0]}
    )


def test_add_default_reparameterisation_w_extra_params(proposal):
    """Test adding a reparameterisation that has extra parameters"""
    proposal.aliases = GWFlowProposal.aliases
    proposal.names = ["ra", "dec"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = []
    proposal.model = MagicMock()
    proposal.model.names = ["ra", "dec"]
    proposal.model.bounds = {
        "ra": [0.0, 2 * np.pi],
        "dec": [-np.pi / 2, np.pi / 2],
    }

    def add_parameters(*args):
        proposal._reparameterisation.parameters = ["ra", "dec"]

    proposal._reparameterisation.add_reparameterisation = MagicMock(
        side_effect=add_parameters
    )

    reparam = MagicMock()
    reparam.__name__ = "MockReparam"
    reparam.parameters = ["ra", "dec"]
    with patch(
        "nessai.gw.proposal.get_gw_reparameterisation",
        return_value=(reparam, {}),
    ) as mock_get:
        GWFlowProposal.add_default_reparameterisations(proposal)

    # Mustn't be called twice despite have two parameters.
    mock_get.assert_called_once_with("sky-ra-dec")
    reparam.assert_called_once_with(
        parameters=["ra", "dec"],
        prior_bounds={
            "ra": [0.0, 2.0 * np.pi],
            "dec": [-np.pi / 2, np.pi / 2],
        },
    )


def test_add_default_reparameterisation_unknown(proposal):
    """Test the method with an unknown parameters"""
    proposal.aliases = GWFlowProposal.aliases
    proposal.names = ["x"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = []
    proposal.model = MagicMock()
    proposal.model.names = ["x"]
    proposal.model.bounds = {"x": [0, 10]}

    with patch(
        "nessai.gw.proposal.get_gw_reparameterisation",
        return_value=(None, {}),
    ) as mock_get:
        GWFlowProposal.add_default_reparameterisations(proposal)

    mock_get.assert_not_called()


def test_augmented_get_reparameterisation(augmented_proposal):
    """Test to make sure the correct version of get reparameterisation is
    called.
    """
    with patch(
        "nessai.gw.proposal.get_gw_reparameterisation", return_value="reparam"
    ) as mock:
        out = AugmentedGWFlowProposal.get_reparameterisation(
            augmented_proposal, "default"
        )
    assert out == "reparam"
    mock.assert_called_once_with("default")


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

    with patch("nessai.proposal.flowproposal.FlowProposal.log_prior") as mock:
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

    with patch(
        "nessai.proposal.flowproposal.FlowProposal." "x_prime_log_prior"
    ) as mock:
        AugmentedGWFlowProposal.x_prime_log_prior(augmented_proposal, 1)

    augmented_proposal.augmented_prior.assert_called_once_with(1)
    mock.assert_called_once_with(1)


@pytest.mark.integration_test
def test_default_reparameterisations(caplog, tmpdir):
    """Assert that the GW defaults are used even in reparameterisations
    is not given.
    """
    caplog.set_level("INFO")
    model = MagicMock()
    model.names = ["mass_ratio", "theta_jn", "phase"]
    model.bounds = {
        "mass_ratio": [20.0, 40.0],
        "theta_jn": [0.0, pi],
        "phase": [0.0, 2 * pi],
    }
    model.reparameterisations = None
    expected_params = model.names + ["phase_radial"]
    proposal = GWFlowProposal(
        model, poolsize=100, output=str(tmpdir.mkdir("test"))
    )
    # Mocked model so can't verify rescaling
    proposal.verify_rescaling = MagicMock()
    proposal.initialise()
    assert proposal._reparameterisation is not None
    assert all(
        [p in expected_params for p in proposal._reparameterisation.parameters]
    )
    assert "reparameterisations included in GWFlowProposal" in str(caplog.text)
