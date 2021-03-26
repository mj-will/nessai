# -*- coding: utf-8 -*-
"""Test general configuration functions"""
import pytest
from unittest.mock import create_autospec

from nessai.proposal import FlowProposal


@pytest.fixture()
def proposal():
    return create_autospec(FlowProposal)


def test_config_drawsize_none(proposal):
    """Test the popluation configuration with no drawsize given"""
    FlowProposal.configure_population(proposal, 2000, None, True, 10, 1.0, 0.0,
                                      'gaussian')
    assert proposal.drawsize == 2000


def test_config_poolsize_none(proposal):
    """
    Test the popluation configuration raises an error if poolsize is None.
    """
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_population(
            proposal, None, None, True, 10, 1.0, 0.0, 'gaussian')

    assert 'poolsize' in str(excinfo.value)


@pytest.mark.parametrize('fixed_radius', [False, 5.0, 1])
def test_config_fixed_radius(proposal, fixed_radius):
    """Test the configuration for a fixed radius"""
    FlowProposal.configure_fixed_radius(proposal, fixed_radius)
    assert proposal.fixed_radius == fixed_radius


def test_config_fixed_radius_not_float(proposal):
    """
    Test the fixed radius is disabled when the radius cannot be converted to
    a float.
    """
    FlowProposal.configure_fixed_radius(proposal, None)
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


@pytest.mark.parametrize("rmin, rmax", [(None, 1.0), (1.0, '2')])
def test_min_max_radius_invalid_input(proposal, rmin, rmax):
    """Test configuration of min radius and no max radius"""
    with pytest.raises(RuntimeError):
        FlowProposal.configure_min_max_radius(proposal, rmin, rmax)


def test_init(proposal, model):
    """Test init with the default parameters"""
    fp = FlowProposal(model, poolsize=1000, kwargs={'priors': 'uniform'})
    assert fp._poolsize == 1000
    assert fp.flow_config is not None
