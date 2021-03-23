# -*- coding: utf-8 -*-
"""Test general configuration functions"""
import pytest

from nessai.proposal import FlowProposal


@pytest.fixture()
def proposal(model):
    return FlowProposal(model, poolsize=1000)


def test_config_drawsize_none(proposal):
    """Test the popluation configuration with no drawsize given"""
    proposal.configure_population(2000, None, True, 10, 1.0, 0.0, 'gaussian')
    assert proposal.drawsize == 2000


def test_config_poolsize_none(proposal):
    """
    Test the popluation configuration raises an error if poolsize is None.
    """
    with pytest.raises(RuntimeError) as excinfo:
        proposal.configure_population(
            None, None, True, 10, 1.0, 0.0, 'gaussian')

    assert 'poolsize' in str(excinfo.value)


@pytest.mark.parametrize('fixed_radius', [False, 5.0, 1])
def test_config_fixed_radius(proposal, fixed_radius):
    """Test the configuration for a fixed radius"""
    proposal.configure_fixed_radius(fixed_radius)
    assert proposal.fixed_radius == fixed_radius


def test_config_fixed_radius_not_float(proposal):
    """
    Test the fixed radius is disabled when the radius cannot be converted to
    a float.
    """
    proposal.configure_fixed_radius(None)
    assert proposal.fixed_radius is False
