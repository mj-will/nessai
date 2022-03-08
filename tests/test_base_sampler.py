# -*- coding: utf-8 -*-
"""Test the base nested sampler"""
import datetime
import os
import pickle
import pytest
import time
from unittest.mock import MagicMock, create_autospec, patch

from nessai.basesampler import BaseNestedSampler


@pytest.fixture
def sampler():
    obj = create_autospec(BaseNestedSampler)
    obj.model = MagicMock()
    return obj


def test_init(sampler):
    """Assert the init sets the correct attributes and calls the correct
    methods.
    """
    model = MagicMock()
    model.configure_pool = MagicMock()
    model.verify_model = MagicMock()
    nlive = 100
    output = './'
    seed = 190425
    checkpointing = True
    resume_file = 'test.pkl'
    plot = False
    n_pool = 2
    pool = MagicMock()

    sampler.configure_random_seed = MagicMock()
    sampler.configure_output = MagicMock()

    BaseNestedSampler.__init__(
        sampler,
        model,
        nlive,
        output=output,
        seed=seed,
        checkpointing=checkpointing,
        resume_file=resume_file,
        plot=plot,
        n_pool=n_pool,
        pool=pool,
    )

    model.verify_model.assert_called_once()
    model.configure_pool.assert_called_once_with(pool=pool, n_pool=n_pool)

    sampler.configure_random_seed.assert_called_once_with(seed)
    sampler.configure_output.assert_called_once_with(
        output, resume_file=resume_file
    )

    assert sampler.nlive == nlive
    assert sampler.plot == plot
    assert sampler.checkpointing == checkpointing
    assert sampler.live_points is None
    assert sampler.iteration == 0
    assert sampler.finalised is False


def test_likelihood_evaluation_time(sampler):
    time = datetime.timedelta(seconds=4)
    sampler.model.likelihood_evaluation_time = time
    out = BaseNestedSampler.likelihood_evaluation_time.__get__(sampler)
    assert out is time


def test_current_sampling_time(sampler):
    """Test the current sampling time"""
    sampler.finalised = False
    sampler.sampling_time = datetime.timedelta(seconds=10)
    sampler.sampling_start_time = datetime.datetime.now()
    time.sleep(0.01)
    t = BaseNestedSampler.current_sampling_time.__get__(sampler)
    assert t.total_seconds() > 10.


def test_current_sampling_time_finalised(sampler):
    """Test the current sampling time if the sampling has been finalised"""
    sampler.finalised = True
    sampler.sampling_time = 10
    assert BaseNestedSampler.current_sampling_time.__get__(sampler) == 10


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


def test_configure_output_none(sampler, tmpdir):
    """Test setting up the output directories if the output is None"""
    p = tmpdir.mkdir('outputs')
    sampler.plot = False
    with patch('os.getcwd', return_value=str(f'{p}/test_cwd/')) as mock:
        BaseNestedSampler.configure_output(sampler, None)

    mock.assert_called_once()
    assert sampler.output == f'{p}/test_cwd/'


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


def test_nested_sampling_loop(sampler):
    """Assert an error is raised"""
    with pytest.raises(NotImplementedError):
        BaseNestedSampler.nested_sampling_loop(sampler)


def test_close_pool(sampler):
    """Assert the method in the model is called with the correct code"""
    sampler.model.close_pool = MagicMock()
    BaseNestedSampler.close_pool(sampler, 2)
    sampler.model.close_pool.assert_called_once_with(code=2)


def test_resume(model):
    """Test the resume method"""
    obj = MagicMock()
    obj.model = None
    obj._previous_likelihood_evaluations = 3
    obj._previous_likelihood_evaluation_time = 4.0

    model.likelihood_evaluations = 1
    model.likelihood_evaluation_time = datetime.timedelta(seconds=2)

    with patch('pickle.load', return_value=obj) as mock_pickle, \
         patch('builtins.open'):
        out = BaseNestedSampler.resume('test.pkl', model)

    mock_pickle.assert_called_once()

    assert out.model == model
    assert out.model.likelihood_evaluations == 4
    assert out.model.likelihood_evaluation_time.total_seconds() == 6


def test_get_result_dictionary(sampler):
    """Assert the correct dictionary is returned"""
    sampler.seed = 170817
    sampler.sampling_time = datetime.timedelta(seconds=4)
    sampler.likelihood_evaluation_time = datetime.timedelta(seconds=2)
    sampler.model.truth = 1.0
    sampler.model.likelihood_evaluations = 10

    d = BaseNestedSampler.get_result_dictionary(sampler)

    assert d['seed'] == sampler.seed
    assert d['sampling_time'] == 4
    assert d['total_likelihood_evaluations'] == 10
    assert d['likelihood_evaluation_time'] == 2
    assert d['truth'] == 1.0


def test_getstate(sampler):
    """Assert the model is deleted and the evaluations are storred"""
    sampler.model.likelihood_evaluations = 10
    sampler.model.likelihood_evaluation_time = datetime.timedelta(seconds=4)
    d = BaseNestedSampler.__getstate__(sampler)
    assert 'model' not in d
    assert d['_previous_likelihood_evaluations'] == 10
    assert d['_previous_likelihood_evaluation_time'] == 4
