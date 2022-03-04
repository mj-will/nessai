# -*- coding: utf-8 -*-
"""
Test general config functions called when the nested sampler is initialised.
"""
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from nessai.nestedsampler import NestedSampler


def test_init(sampler):
    """Test the init method"""
    pool = 'pool'
    n_pool = 2
    model = MagicMock()
    model.verify_model = MagicMock()
    model.configure_pool = MagicMock()

    sampler.setup_output = MagicMock()
    sampler.configure_flow_reset = MagicMock()
    sampler.configure_flow_proposal = MagicMock()
    sampler.configure_uninformed_proposal = MagicMock
    NestedSampler.__init__(
        sampler,
        model,
        nlive=100,
        poolsize=100,
        pool=pool,
        n_pool=n_pool,
    )
    assert sampler.initialised is False
    assert sampler.nlive == 100
    model.verify_model.assert_called_once()
    model.configure_pool.assert_called_once_with(pool=pool, n_pool=n_pool)


def test_setup_output_w_plotting(sampler, tmpdir):
    """Test setting up the output directories with plot=True"""
    p = tmpdir.mkdir('outputs')
    sampler.plot = True
    with patch(
        'nessai.nestedsampler.BaseNestedSampler.configure_output'
    ) as mock:
        NestedSampler.configure_output(sampler, f'{p}/tests', 'resume.pkl')
    mock.assert_called_once_with(f'{p}/tests', 'resume.pkl')
    assert os.path.exists(f'{p}/tests/diagnostics')


def test_configure_max_iteration(sampler):
    """Test to make sure the maximum iteration is set correctly"""
    NestedSampler.configure_max_iteration(sampler, 10)
    assert sampler.max_iteration == 10


def test_configure_no_max_iteration(sampler):
    """Test to make sure if no max iteration is given it is set to inf"""
    NestedSampler.configure_max_iteration(sampler, None)
    assert sampler.max_iteration == np.inf


def test_training_frequency(sampler):
    """Make sure training frequency is set"""
    NestedSampler.configure_training_frequency(sampler, 100)
    assert sampler.training_frequency == 100


@pytest.mark.parametrize('f', [None, 'inf', 'None'])
def test_training_frequency_on_empty(sampler, f):
    """Test the values that should give 'train on empty'"""
    NestedSampler.configure_training_frequency(sampler, f)
    assert sampler.training_frequency == np.inf
