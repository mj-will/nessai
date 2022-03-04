# -*- coding: utf-8 -*-
"""Test the base nested sampler"""
import datetime
import os
import pickle
import pytest
from unittest.mock import MagicMock, create_autospec, patch

from nessai.basesampler import BaseNestedSampler


@pytest.fixture
def sampler():
    return create_autospec(BaseNestedSampler)


@patch('numpy.random.seed')
@patch('torch.manual_seed')
def test_set_random_seed(mock1, mock2, sampler):
    """Test the correct functions are called when setting the random seed"""
    BaseNestedSampler.configure_random_seed(sampler, 150914)
    mock1.assert_called_once_with(150914)
    mock2.assert_called_once_with(seed=150914)


@patch('numpy.random.seed')
@patch('torch.manual_seed')
def test_no_random_seed(mock1, mock2, sampler):
    """Assert no seed is set if seed=None"""
    BaseNestedSampler.configure_random_seed(sampler, None)
    mock1.assert_not_called()
    mock2.assert_not_called()


def test_configure_output(sampler, tmpdir):
    """Test setting up the output directories"""
    p = tmpdir.mkdir('outputs')
    sampler.plot = False
    BaseNestedSampler.configure_output(sampler, f'{p}/tests')
    assert os.path.exists(f'{p}/tests')
    assert sampler.resume_file == f'{p}/tests/nested_sampler_resume.pkl'


def test_configure_output_w_resume(sampler, tmpdir):
    """Test output configuration with a specified resume file"""
    p = tmpdir.mkdir('outputs')
    sampler.plot = False
    BaseNestedSampler.configure_output(sampler, f'{p}/tests', 'resume.pkl')
    assert sampler.resume_file == f'{p}/tests/resume.pkl'


@pytest.mark.parametrize('periodic', [False, True])
def test_checkpoint(sampler, periodic):
    """Test checkpointing method.

    Make sure a file is produced and that the sampling time is updated.
    Also checks to make sure that the iteration is recorded when periodic=False
    """
    sampler.checkpoint_iterations = [10]
    sampler.iteration = 20
    now = datetime.datetime.now()
    sampler.sampling_start_time = now
    sampler.sampling_time = datetime.timedelta()
    sampler.resume_file = 'test.pkl'

    with patch('nessai.basesampler.safe_file_dump') as sfd_mock:
        BaseNestedSampler.checkpoint(sampler, periodic=periodic)

    sfd_mock.assert_called_once_with(
        sampler, sampler.resume_file, pickle, save_existing=True
    )

    assert sampler.sampling_start_time > now
    assert sampler.sampling_time.total_seconds() > 0.

    if periodic:
        assert sampler.checkpoint_iterations == [10]
    else:
        assert sampler.checkpoint_iterations == [10, 20]


def test_resume(model):
    """Test the resume method"""
    obj = MagicMock()
    obj.model = None
    obj._previous_likelihood_evaluations = 3
    model.likelihood_evaluations = 1

    with patch('pickle.load', return_value=obj) as mock_pickle, \
         patch('builtins.open'):
        out = BaseNestedSampler.resume('test.pkl', model)

    mock_pickle.assert_called_once()

    assert out.model == model
    assert out.model.likelihood_evaluations == 4
