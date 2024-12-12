# -*- coding: utf-8 -*-
"""Test the base nested sampler"""

import copy
import datetime
import os
import pickle
import time
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest

from nessai.samplers.base import BaseNestedSampler


@pytest.fixture
def sampler(rng):
    obj = create_autospec(BaseNestedSampler)
    obj.model = MagicMock()
    obj.rng = rng
    return obj


@pytest.mark.parametrize("checkpoint_on_iteration", [False, True])
def test_init(sampler, checkpoint_on_iteration, rng):
    """Assert the init sets the correct attributes and calls the correct
    methods.
    """
    model = MagicMock()
    model.configure_pool = MagicMock()
    model.verify_model = MagicMock()
    nlive = 100
    output = "./"
    seed = 190425
    checkpointing = True
    checkpoint_interval = 10
    log_on_iteration = True
    logging_interval = 10
    resume_file = "test.pkl"
    plot = False
    n_pool = 2
    pool = MagicMock()

    sampler.configure_rng = MagicMock()
    sampler.configure_output = MagicMock()
    sampler.configure_periodic_logging = MagicMock()

    BaseNestedSampler.__init__(
        sampler,
        model,
        nlive,
        output=output,
        seed=seed,
        rng=rng,
        checkpointing=checkpointing,
        checkpoint_interval=checkpoint_interval,
        checkpoint_on_iteration=checkpoint_on_iteration,
        log_on_iteration=log_on_iteration,
        logging_interval=logging_interval,
        resume_file=resume_file,
        plot=plot,
        n_pool=n_pool,
        pool=pool,
    )

    model.verify_model.assert_called_once()
    model.configure_pool.assert_called_once_with(pool=pool, n_pool=n_pool)

    sampler.configure_rng.assert_called_once_with(seed, rng)
    sampler.configure_output.assert_called_once_with(
        output, resume_file=resume_file
    )
    sampler.configure_periodic_logging.assert_called_once_with(
        logging_interval,
        log_on_iteration,
    )

    assert sampler.nlive == nlive
    assert sampler.plot == plot
    assert sampler.checkpointing == checkpointing
    assert sampler.checkpoint_interval == checkpoint_interval
    assert sampler.checkpoint_on_iteration == checkpoint_on_iteration
    assert sampler.live_points is None
    assert sampler.iteration == 0
    assert sampler.finalised is False
    assert sampler.history is None


def test_likelihood_evaluation_time(sampler):
    time = datetime.timedelta(seconds=4)
    sampler.model.likelihood_evaluation_time = time
    out = BaseNestedSampler.likelihood_evaluation_time.__get__(sampler)
    assert out is time


def test_current_sampling_time(sampler, wait):
    """Test the current sampling time"""
    sampler.finalised = False
    sampler.sampling_time = datetime.timedelta(seconds=10)
    sampler.sampling_start_time = datetime.datetime.now()
    wait()
    t = BaseNestedSampler.current_sampling_time.__get__(sampler)
    assert t.total_seconds() > 10.0


def test_total_likelihood_evaluations(sampler):
    """Check likelihood calls from model are returned"""
    sampler.model = MagicMock()
    sampler.model.likelihood_evaluations = 10
    assert (
        BaseNestedSampler.total_likelihood_evaluations.__get__(sampler) == 10
    )


def test_likelihood_calls(sampler):
    """Check likelihood calls from model are returned"""
    sampler.model = MagicMock()
    sampler.model.likelihood_evaluations = 10
    assert BaseNestedSampler.likelihood_calls.__get__(sampler) == 10


def test_current_sampling_time_finalised(sampler):
    """Test the current sampling time if the sampling has been finalised"""
    sampler.finalised = True
    sampler.sampling_time = 10
    assert BaseNestedSampler.current_sampling_time.__get__(sampler) == 10


def test_posterior_effective_sample_suze(sampler):
    """Assert an error is raised"""
    with pytest.raises(NotImplementedError):
        BaseNestedSampler.posterior_effective_sample_size.__get__(sampler)


def test_set_random_seed(sampler, rng):
    """Test the correct functions are called when setting the random seed"""
    with (
        patch("numpy.random.default_rng", return_value=rng) as mock_rng,
        patch("torch.manual_seed") as mock_torch_seed,
    ):
        BaseNestedSampler.configure_rng(sampler, 150914, None)
    mock_rng.assert_called_once_with(150914)
    mock_torch_seed.assert_called_once_with(150914)
    assert sampler.seed == 150914
    assert sampler.rng is rng


def test_no_random_seed_or_rng(sampler, rng):
    """Test behaviour when no seed or rng is provided"""
    with (
        patch("random.randint", return_value=42) as mock_randint,
        patch(
            "numpy.random.default_rng", return_value=rng
        ) as mock_default_rng,
        patch("torch.manual_seed") as mock_torch_seed,
    ):
        BaseNestedSampler.configure_rng(sampler, None, None)
    mock_randint.assert_called_once()
    mock_default_rng.assert_called_once_with(42)
    mock_torch_seed.assert_called_once_with(42)
    assert sampler.seed == 42
    assert sampler.rng is rng


def test_random_seed_rng_only(sampler):
    """Test behaviour when no seed or rng is provided"""
    rng = MagicMock()
    rng.integers = MagicMock(return_value=42)
    with (
        patch("numpy.random.default_rng") as mock_default_rng,
        patch("torch.manual_seed") as mock_torch_seed,
    ):
        BaseNestedSampler.configure_rng(sampler, rng=rng)
    mock_default_rng.assert_not_called()
    rng.integers.assert_called_once()
    mock_torch_seed.assert_called_once_with(42)
    assert sampler.seed == 42


@pytest.mark.integration_test
def test_configure_rng_rng_integration(sampler):
    rng = np.random.default_rng(42)
    BaseNestedSampler.configure_rng(sampler, rng=rng)
    orig_seed = copy.copy(sampler.seed)
    rng = np.random.default_rng(42)
    BaseNestedSampler.configure_rng(sampler, rng=rng)
    assert sampler.seed == orig_seed


@pytest.mark.integration_test
def configure_rng_no_rng_integration(sampler):
    """Ensure a run without a seed can be reproduced using the seed that
    is generated.
    """
    BaseNestedSampler.configure_rng(sampler)
    orig_seed = copy.copy(sampler.seed)
    x = sampler.rng.integers(0, 100)
    BaseNestedSampler.configure_rng(sampler, seed=orig_seed)
    x_re = sampler.rng.integers(0, 100)
    assert x == x_re


def test_configure_output(sampler, tmpdir):
    """Test setting up the output directories"""
    p = tmpdir.mkdir("outputs")
    sampler.plot = False
    path = os.path.join(p, "tests")
    BaseNestedSampler.configure_output(sampler, path)
    assert os.path.exists(path)
    assert sampler.resume_file == os.path.join(
        path, "nested_sampler_resume.pkl"
    )


def test_configure_output_none(sampler, tmpdir):
    """Test setting up the output directories if the output is None"""
    p = tmpdir.mkdir("outputs")
    sampler.plot = False
    path = os.path.join(p, "test_cwd")
    with patch("os.getcwd", return_value=path) as mock:
        BaseNestedSampler.configure_output(sampler, None)

    mock.assert_called_once()
    assert sampler.output == path


def test_configure_output_w_resume(sampler, tmpdir):
    """Test output configuration with a specified resume file"""
    p = tmpdir.mkdir("outputs")
    sampler.plot = False
    path = os.path.join(p, "tests")
    BaseNestedSampler.configure_output(sampler, path, "resume.pkl")
    assert sampler.resume_file == os.path.join(path, "resume.pkl")


def test_configure_periodic_logging_time(sampler):
    """Assert the perdiodic logging is correctly configured"""
    with patch("time.time", return_value=10) as mock:
        BaseNestedSampler.configure_periodic_logging(
            sampler,
            20,
            False,
        )
    mock.assert_called_once()
    assert sampler._last_log == 10
    assert sampler.logging_interval == 20
    assert sampler.log_on_iteration is False


@pytest.mark.parametrize("interval, expected", [(50, 50), (None, 100)])
def test_configure_periodic_logging_interval(sampler, interval, expected):
    """Assert the perdiodic logging is correctly configured"""
    sampler.nlive = 100
    BaseNestedSampler.configure_periodic_logging(
        sampler,
        interval,
        True,
    )
    assert sampler._last_log == 0
    assert sampler.logging_interval == expected
    assert sampler.log_on_iteration is True


def test_configure_periodic_logging_all_false(sampler):
    """Assert log on iteration is enabled if both inputs are false"""
    sampler.nlive = 100
    BaseNestedSampler.configure_periodic_logging(sampler, None, False)
    assert sampler.log_on_iteration is True
    assert sampler._last_log == 0
    assert sampler.logging_interval == 100


def test_log_state(sampler):
    """Assert a NotImplementedError is raised"""
    with pytest.raises(NotImplementedError):
        BaseNestedSampler.log_state(sampler)


@pytest.mark.parametrize("interval", [20, 40])
def test_periodically_log_state_time(sampler, interval):
    """Assert log_state is called sufficient time has elapsed.

    Set such that 30 seconds have elapsed since the last log.
    """
    now = time.time()
    elapsed = 30
    sampler._last_log = now - elapsed
    sampler.logging_interval = interval
    sampler.log_on_iteration = False
    with patch("time.time", return_value=now):
        BaseNestedSampler.periodically_log_state(sampler)
    # if more time has passed, then should log
    if elapsed > interval:
        sampler.log_state.assert_called_once()
        assert sampler._last_log == now
    else:
        sampler.log_state.assert_not_called()
        assert sampler._last_log == (now - elapsed)


@pytest.mark.parametrize("interval", [20, 40])
def test_periodically_log_state_iteration(sampler, interval):
    """Assert log_state is called sufficient iterations have elapsed.

    Set such that 30 iterations have elapsed since the last log.
    """
    elapsed = 30
    sampler.iteration = 120
    sampler._last_log = sampler.iteration - elapsed
    sampler.logging_interval = interval
    sampler.log_on_iteration = True
    BaseNestedSampler.periodically_log_state(sampler)
    # if more iterations have passed, then should log
    if elapsed > interval:
        sampler.log_state.assert_called_once()
        assert sampler._last_log == sampler.iteration
    else:
        sampler.log_state.assert_not_called()
        assert sampler._last_log == (sampler.iteration - elapsed)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("no_history", [False, True])
def test_checkpoint_iteration(sampler, wait, periodic, no_history):
    """Test checkpointing method on iterations.

    Make sure a file is produced and that the sampling time is updated.
    Also checks to make sure that the iteration is recorded when periodic=False
    """
    if no_history:
        sampler.history = None
    else:
        sampler.history = dict(checkpoint_iterations=[10])
    sampler.checkpoint_on_iteration = True
    sampler.checkpoint_interval = 10
    sampler.checkpoint_callback = None
    sampler._last_checkpoint = 0
    sampler.iteration = 20
    now = datetime.datetime.now()
    sampler.sampling_start_time = now
    sampler.sampling_time = datetime.timedelta()
    sampler.resume_file = "test.pkl"

    with patch("nessai.samplers.base.safe_file_dump") as sfd_mock:
        wait()
        BaseNestedSampler.checkpoint(sampler, periodic=periodic)

    sfd_mock.assert_called_once_with(
        sampler, sampler.resume_file, pickle, save_existing=True
    )

    assert sampler.sampling_start_time > now
    assert sampler.sampling_time.total_seconds() > 0.0

    if periodic:
        if not no_history:
            assert sampler.history["checkpoint_iterations"] == [10]
        else:
            assert sampler.history is None
        assert sampler._last_checkpoint == 20
    else:
        if not no_history:
            assert sampler.history["checkpoint_iterations"] == [10, 20]
        else:
            assert sampler.history is None


def test_checkpoint_time(sampler, wait):
    """Test checkpointing method based on time interval

    Make sure a file is produced and that the sampling time is updated.
    """
    now = datetime.datetime.now()
    sampler.checkpoint_iterations = [10]
    sampler.checkpoint_on_iteration = False
    sampler.checkpoint_interval = 15 * 60
    sampler.checkpoint_callback = None
    sampler.sampling_start_time = now - datetime.timedelta(minutes=32)
    sampler._last_checkpoint = now - datetime.timedelta(minutes=16)
    sampler.iteration = 20
    sampler.sampling_time = datetime.timedelta()
    sampler.resume_file = "test.pkl"

    with (
        patch("nessai.samplers.base.safe_file_dump") as sfd_mock,
        patch("datetime.datetime", return_value=now) as mock_datetime,
    ):
        wait()
        mock_datetime.now.return_value = now
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(
            *args, **kw
        )
        BaseNestedSampler.checkpoint(sampler, periodic=True)

    sfd_mock.assert_called_once_with(
        sampler, sampler.resume_file, pickle, save_existing=True
    )
    assert sampler._last_checkpoint is now


def test_checkpoint_periodic_skipped_iteration(sampler):
    """Assert the sampler does not checkpoint if the criterion is not met"""
    sampler.checkpoint_on_iteration = True
    sampler.iteration = 10
    sampler._last_checkpoint = 9
    sampler.checkpoint_interval = 10
    sampler.checkpoint_callback = None
    with patch("nessai.samplers.base.safe_file_dump") as sfd_mock:
        BaseNestedSampler.checkpoint(sampler, periodic=True)
    sfd_mock.assert_not_called()


def test_checkpoint_periodic_skipped_time(sampler):
    """Assert the sampler does not checkpoint if the criterion is not met"""
    sampler.checkpoint_on_iteration = False
    sampler.iteration = 10
    sampler._last_checkpoint = datetime.datetime.now()
    sampler.checkpoint_interval = 600
    sampler.checkpoint_callback = None
    with patch("nessai.samplers.base.safe_file_dump") as sfd_mock:
        BaseNestedSampler.checkpoint(sampler, periodic=True)
    sfd_mock.assert_not_called()


def test_checkpoint_force(sampler):
    """Assert the sampler checkpoints if force=True"""
    now = datetime.datetime.now()
    sampler.sampling_start_time = now - datetime.timedelta(minutes=32)
    sampler.sampling_time = datetime.timedelta()
    sampler.resume_file = "test.pkl"
    sampler.checkpoint_callback = None
    with patch("nessai.samplers.base.safe_file_dump") as sfd_mock:
        BaseNestedSampler.checkpoint(sampler, periodic=True, force=True)
    sfd_mock.assert_called_once_with(
        sampler, sampler.resume_file, pickle, save_existing=True
    )


def test_checkpoint_callback(sampler):
    """Assert the checkpoint callback is used"""

    callback = MagicMock()

    sampler.history = dict(checkpoint_iterations=[10])
    sampler.checkpoint_on_iteration = True
    sampler.checkpoint_interval = 10
    sampler.checkpoint_callback = callback
    sampler._last_checkpoint = 0
    sampler.iteration = 20
    now = datetime.datetime.now()
    sampler.sampling_start_time = now
    sampler.sampling_time = datetime.timedelta()

    BaseNestedSampler.checkpoint(sampler)
    callback.assert_called_once_with(sampler)


def test_nested_sampling_loop(sampler):
    """Assert an error is raised"""
    with pytest.raises(NotImplementedError):
        BaseNestedSampler.nested_sampling_loop(sampler)


def test_close_pool(sampler):
    """Assert the method in the model is called with the correct code"""
    sampler.model.close_pool = MagicMock()
    BaseNestedSampler.close_pool(sampler, 2)
    sampler.model.close_pool.assert_called_once_with(code=2)


@pytest.mark.parametrize("output", [None, "orig", "new"])
def test_resume_from_pickled_sampler(model, output, rng):
    """Test the resume from pickled sampler method"""
    obj = MagicMock()
    obj.model = None
    obj.output = "orig"
    obj.rng = rng
    obj._previous_likelihood_evaluations = 3
    obj._previous_likelihood_evaluation_time = 4.0

    model.rng = None
    model.set_rng = MagicMock()
    model.likelihood_evaluations = 1
    model.likelihood_evaluation_time = datetime.timedelta(seconds=2)

    out = BaseNestedSampler.resume_from_pickled_sampler(
        obj, model, output=output
    )

    assert out.model == model
    assert out.model.likelihood_evaluations == 4
    assert out.model.likelihood_evaluation_time.total_seconds() == 6
    model.set_rng.assert_called_once_with(rng)

    if output == "new":
        obj.update_output.assert_called_once_with("new")
    else:
        obj.update_output.assert_not_called()


def test_resume(model):
    """Test the resume method"""
    obj = MagicMock()
    pickle_out = MagicMock()

    with (
        patch("pickle.load", return_value=obj) as mock_pickle,
        patch("builtins.open"),
        patch(
            "nessai.samplers.base.BaseNestedSampler.resume_from_pickled_sampler",
            return_value=pickle_out,
        ) as mock_resume,
    ):
        out = BaseNestedSampler.resume("test.pkl", model, output="test")

    assert out is pickle_out
    mock_pickle.assert_called_once()
    mock_resume.assert_called_once_with(obj, model, output="test")


def test_get_result_dictionary(sampler):
    """Assert the correct dictionary is returned"""
    sampler.seed = 170817
    sampler.sampling_time = datetime.timedelta(seconds=4)
    sampler.likelihood_evaluation_time = datetime.timedelta(seconds=2)
    sampler.model.truth = 1.0
    sampler.model.likelihood_evaluations = 10
    sampler.history = None

    d = BaseNestedSampler.get_result_dictionary(sampler)

    assert d["seed"] == sampler.seed
    assert d["sampling_time"] == 4
    assert d["total_likelihood_evaluations"] == 10
    assert d["likelihood_evaluation_time"] == 2
    assert d["truth"] == 1.0
    assert "history" in d


def test_getstate(sampler):
    """Assert the model is deleted and the evaluations are storred"""
    sampler.model.likelihood_evaluations = 10
    sampler.model.likelihood_evaluation_time = datetime.timedelta(seconds=4)
    d = BaseNestedSampler.__getstate__(sampler)
    assert "model" not in d
    assert d["_previous_likelihood_evaluations"] == 10
    assert d["_previous_likelihood_evaluation_time"] == 4


def test_initialise_history(sampler):
    sampler.history = None
    BaseNestedSampler.initialise_history(sampler)
    assert isinstance(sampler.history, dict)
    assert "sampling_time" in sampler.history
    assert "likelihood_evaluations" in sampler.history


def test_initialise_history_skip(sampler, caplog):
    caplog.set_level("DEBUG")
    sampler.history = {}
    BaseNestedSampler.initialise_history(sampler)
    assert "already initialised" in str(caplog.text)


def test_update_history(sampler):
    sampler.total_likelihood_evaluations = 20
    sampler.current_sampling_time = datetime.timedelta(seconds=2)
    sampler.history = dict(likelihood_evaluations=[10], sampling_time=[1])
    BaseNestedSampler.update_history(sampler)
    assert sampler.history["likelihood_evaluations"] == [10, 20]
    assert sampler.history["sampling_time"] == [1, 2]


def test_update_output(sampler, tmp_path):
    sampler.output = tmp_path / "orig"
    sampler.resume_file = sampler.output / "resume.pkl"
    new_output = tmp_path / "new"
    BaseNestedSampler.update_output(sampler, new_output)
    assert sampler.output == new_output
    assert sampler.resume_file == str(new_output / "resume.pkl")
