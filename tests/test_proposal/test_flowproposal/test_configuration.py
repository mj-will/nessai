# -*- coding: utf-8 -*-
"""Test general configuration functions"""
import pytest

from nessai.proposal import FlowProposal
from nessai import utils


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
    FlowProposal.configure_fixed_radius(proposal, 'four')
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


@pytest.mark.parametrize('rmin, rmax', [(None, 1.0), (1.0, '2')])
def test_min_max_radius_invalid_input(proposal, rmin, rmax):
    """Test configuration of min radius and no max radius"""
    with pytest.raises(RuntimeError):
        FlowProposal.configure_min_max_radius(proposal, rmin, rmax)


@pytest.mark.parametrize('plot, plot_pool, plot_train',
                         [(True, True, True),
                          ('all', 'all', 'all'),
                          ('train', False, 'all'),
                          ('pool', 'all', False),
                          ('min', True, True),
                          ('minimal', True, True),
                          (False, False, False),
                          ('some', False, False)
                          ])
def test_configure_plotting(proposal, plot, plot_pool, plot_train):
    """Test the configuration of plotting settings"""
    FlowProposal.configure_plotting(proposal, plot)
    assert proposal._plot_pool == plot_pool
    assert proposal._plot_training == plot_train


@pytest.mark.parametrize('latent_prior, prior_func',
                         [('gaussian', 'draw_gaussian'),
                          ('truncated_gaussian', 'draw_truncated_gaussian'),
                          ('uniform', 'draw_uniform'),
                          ('uniform_nsphere', 'draw_nsphere'),
                          ('uniform_nball', 'draw_nsphere')
                          ])
def test_configure_latent_prior(proposal, latent_prior, prior_func):
    """Test to make sure the correct latent priors are used."""
    proposal.latent_prior = latent_prior
    proposal.flow_config = {'model_config': {}}
    FlowProposal.configure_latent_prior(proposal)
    assert proposal.draw_latent_prior == getattr(utils, prior_func)


def test_configure_latent_prior_var(proposal):
    """
    Test the latent prior when using a truncated Gaussian with a variance.
    """
    proposal.latent_prior = 'truncated_gaussian'
    proposal.flow_config = {'model_config': {'kwargs': {'var': 4}}}
    proposal.draw_latent_kwargs = {}
    FlowProposal.configure_latent_prior(proposal)
    assert proposal.draw_latent_prior == \
        getattr(utils, 'draw_truncated_gaussian')
    assert proposal.draw_latent_kwargs.get('var') == 4


def test_configure_latent_prior_unknown(proposal):
    """Make sure unknown latent priors raise an error"""
    proposal.latent_prior = 'truncated'
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_latent_prior(proposal)
    assert 'Unknown latent prior: truncated, ' in str(excinfo.value)
