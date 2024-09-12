# -*- coding: utf-8 -*-
"""Test general configuration functions"""

from unittest.mock import patch

import pytest

from nessai import utils
from nessai.proposal import FlowProposal


def test_config_drawsize_none(proposal):
    """Test the popluation configuration with no drawsize given"""
    proposal.poolsize = 2000
    FlowProposal.configure_population(proposal, None, 1.0, 0.0, "gaussian")
    assert proposal.drawsize == 2000


@pytest.mark.parametrize("fixed_radius", [False, 5.0, 1])
def test_config_fixed_radius(proposal, fixed_radius):
    """Test the configuration for a fixed radius"""
    FlowProposal.configure_fixed_radius(proposal, fixed_radius)
    assert proposal.fixed_radius == fixed_radius


def test_config_fixed_radius_not_float(proposal):
    """
    Test the fixed radius is disabled when the radius cannot be converted to
    a float.
    """
    FlowProposal.configure_fixed_radius(proposal, "four")
    assert proposal.fixed_radius is False


def test_min_radius_no_max(proposal):
    """Test configuration of min radius and no max radius"""
    FlowProposal.configure_min_max_radius(proposal, 5.0, False)
    assert proposal.min_radius == 5.0
    assert proposal.max_radius is False


def test_min_max_radius(proposal):
    """Test configuration of min radius and no max radius"""
    FlowProposal.configure_min_max_radius(proposal, 5, 10)
    assert proposal.min_radius == 5.0
    assert proposal.max_radius == 10.0


@pytest.mark.parametrize("rmin, rmax", [(None, 1.0), (1.0, "2")])
def test_min_max_radius_invalid_input(proposal, rmin, rmax):
    """Test configuration of min radius and no max radius"""
    with pytest.raises(RuntimeError):
        FlowProposal.configure_min_max_radius(proposal, rmin, rmax)


@pytest.mark.parametrize(
    "latent_prior, prior_func",
    [
        ("gaussian", "draw_gaussian"),
        ("truncated_gaussian", "draw_truncated_gaussian"),
        ("uniform", "draw_uniform"),
        ("uniform_nsphere", "draw_nsphere"),
        ("uniform_nball", "draw_nsphere"),
        ("flow", None),
    ],
)
def test_configure_latent_prior(proposal, latent_prior, prior_func):
    """Test to make sure the correct latent priors are used."""
    proposal.latent_prior = latent_prior
    proposal.flow_config = {"model_config": {}}
    FlowProposal.configure_latent_prior(proposal)
    if prior_func:
        assert proposal._draw_latent_prior == getattr(utils, prior_func)
    else:
        assert proposal._draw_latent_prior is None


def test_configure_latent_prior_unknown(proposal):
    """Make sure unknown latent priors raise an error"""
    proposal.latent_prior = "truncated"
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_latent_prior(proposal)
    assert "Unknown latent prior: truncated, " in str(excinfo.value)


@pytest.mark.parametrize(
    "latent_prior",
    ["truncated_gaussian", "uniform_nball", "uniform_nsphere"],
)
def test_configure_constant_volume(proposal, latent_prior):
    """Test configuration for constant volume mode."""
    proposal.constant_volume_mode = True
    proposal.volume_fraction = 0.95
    proposal.rescaled_dims = 5
    proposal.latent_prior = latent_prior
    proposal.max_radius = 3.0
    proposal.min_radius = 5.0
    proposal.fuzz = 1.5
    with patch(
        "nessai.proposal.flowproposal.flowproposal.compute_radius",
        return_value=4.0,
    ) as mock:
        FlowProposal.configure_constant_volume(proposal)
    mock.assert_called_once_with(5, 0.95)
    assert proposal.fixed_radius == 4.0
    assert proposal.min_radius is False
    assert proposal.max_radius is False
    assert proposal.fuzz == 1.0


def test_configure_constant_volume_disabled(proposal):
    """Assert nothing happens if constant_volume is False"""
    proposal.constant_volume_mode = False
    with patch(
        "nessai.proposal.flowproposal.flowproposal.compute_radius"
    ) as mock:
        FlowProposal.configure_constant_volume(proposal)
    mock.assert_not_called()


def test_constant_volume_invalid_latent_prior(proposal):
    """Assert an error is raised if the latent prior is not a truncated \
        Gaussian
    """
    err = "Constant volume mode is not supported for latent_prior=gaussian"
    proposal.constant_volume_mode = True
    proposal.latent_prior = "gaussian"
    with pytest.raises(RuntimeError, match=err):
        FlowProposal.configure_constant_volume(proposal)
