# -*- coding: utf-8 -*-
"""Test different properties in FlowProposal"""
import pytest
from unittest.mock import create_autospec

from nessai.proposal import FlowProposal


@pytest.fixture
def fp_mock():
    return create_autospec(FlowProposal)


def test_poolsize(fp_mock):
    """Test poolsize property"""
    fp_mock._poolsize = 10
    fp_mock._poolsize_scale = 2
    assert FlowProposal.poolsize.__get__(fp_mock) == 20


def test_dims(fp_mock):
    """Test dims property"""
    fp_mock.names = ['x', 'y']
    assert FlowProposal.dims.__get__(fp_mock) == 2


def test_rescaled_dims(fp_mock):
    """Test rescaled_dims property"""
    fp_mock.rescaled_names = ['x', 'y']
    assert FlowProposal.rescaled_dims.__get__(fp_mock) == 2


def test_dtype(fp_mock):
    """Test dims property"""
    fp_mock.names = ['x', 'y']
    fp_mock._x_dtype = None
    assert FlowProposal.x_dtype.__get__(fp_mock) == \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]


def test_prime_dtype(fp_mock):
    """Test dims property"""
    fp_mock.rescaled_names = ['x', 'y']
    fp_mock._x_prime_dtype = None
    assert FlowProposal.x_prime_dtype.__get__(fp_mock) == \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]


def test_population_dtype(fp_mock):
    """Test dims property"""
    fp_mock.x_dtype = \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
    fp_mock.use_x_prime_prior = False
    assert FlowProposal.population_dtype.__get__(fp_mock) == \
        [('x', 'f8'), ('y', 'f8'), ('logP', 'f8'), ('logL', 'f8')]


def test_population_dtype_prime_prior(fp_mock):
    """Test dims property"""
    fp_mock.x_prime_dtype = \
        [('x_p', 'f8'), ('y_p', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
    fp_mock.use_x_prime_prior = True
    assert FlowProposal.population_dtype.__get__(fp_mock) == \
        [('x_p', 'f8'), ('y_p', 'f8'), ('logP', 'f8'), ('logL', 'f8')]
