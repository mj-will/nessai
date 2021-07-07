# -*- coding: utf-8 -*-
"""Test different properties in FlowProposal"""
from nessai.proposal import FlowProposal


def test_poolsize(proposal):
    """Test poolsize property"""
    proposal._poolsize = 10
    proposal._poolsize_scale = 2
    assert FlowProposal.poolsize.__get__(proposal) == 20


def test_dims(proposal):
    """Test dims property"""
    proposal.names = ['x', 'y']
    assert FlowProposal.dims.__get__(proposal) == 2


def test_rescaled_dims(proposal):
    """Test rescaled_dims property"""
    proposal.rescaled_names = ['x', 'y']
    assert FlowProposal.rescaled_dims.__get__(proposal) == 2


def test_flow_names(proposal):
    """Test flow_names property.

    Should be the same as rescaled_names by default
    """
    proposal.rescaled_names = ['x', 'y']
    assert FlowProposal.flow_names.__get__(proposal) == ['x', 'y']


def test_flow_dims(proposal):
    """Test the flow_dims propoety.

    Should be the same as rescaled dims by default
    """
    proposal.rescaled_dims = 2
    assert FlowProposal.flow_dims.__get__(proposal) == 2


def test_dtype(proposal):
    """Test dims property"""
    proposal.names = ['x', 'y']
    proposal._x_dtype = None
    assert FlowProposal.x_dtype.__get__(proposal) == \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]


def test_prime_dtype(proposal):
    """Test dims property"""
    proposal.rescaled_names = ['x', 'y']
    proposal._x_prime_dtype = None
    assert FlowProposal.x_prime_dtype.__get__(proposal) == \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]


def test_population_dtype(proposal):
    """Test dims property"""
    proposal.x_dtype = \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
    proposal.use_x_prime_prior = False
    assert FlowProposal.population_dtype.__get__(proposal) == \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]


def test_population_dtype_prime_prior(proposal):
    """Test dims property"""
    proposal.x_prime_dtype = \
        [('x_p', 'f8'), ('y_p', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
    proposal.use_x_prime_prior = True
    assert FlowProposal.population_dtype.__get__(proposal) == \
        [('x_p', 'f8'), ('y_p', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
