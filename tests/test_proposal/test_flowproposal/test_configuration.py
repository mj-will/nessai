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
