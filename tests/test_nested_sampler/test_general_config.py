# -*- coding: utf-8 -*-
"""
Test general config functions called when the nested sampler is initialised.
"""
import os
from unittest.mock import patch

from nessai.nestedsampler import NestedSampler


@patch('numpy.random.seed')
@patch('torch.manual_seed')
def test_set_random_seed(mock1, mock2, sampler):
    """Test the correct functions are called when setting the random seed"""
    NestedSampler.setup_random_seed(sampler, 150914)
    mock1.assert_called_once_with(150914)
    mock2.assert_called_once_with(seed=150914)


@patch('numpy.random.seed')
@patch('torch.manual_seed')
def test_no_random_seed(mock1, mock2, sampler):
    """Assert no seed is set if seed=None"""
    NestedSampler.setup_random_seed(sampler, None)
    mock1.assert_not_called()
    mock2.assert_not_called()


def test_setup_output(sampler, tmpdir):
    """Test setting up the output directories"""
    tmpdir.mkdir('outputs')
    sampler.plot = False
    resume_file = NestedSampler.setup_output(sampler, 'outputs/tests')
    assert os.path.exists('outputs/tests')
    assert resume_file == 'outputs/tests/nested_sampler_resume.pkl'


def test_setup_output_w_plotting(sampler, tmpdir):
    """Test setting up the output directories with plot=True"""
    tmpdir.mkdir('outputs')
    sampler.plot = True
    NestedSampler.setup_output(sampler, 'outputs/tests')
    assert os.path.exists('outputs/tests/diagnostics')


def test_setup_output_w_resume(sampler, tmpdir):
    """Test output configuration with a specified resume file"""
    tmpdir.mkdir('outputs')
    sampler.plot = False
    resume_file = \
        NestedSampler.setup_output(sampler, 'outputs/tests', 'resume.pkl')
    assert resume_file == 'outputs/tests/resume.pkl'
