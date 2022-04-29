# -*- coding: utf-8 -*-
"""
Tests for the FlowSampler class.
"""
import os
import signal
import sys
import time
from threading import Thread

import pytest
from nessai.flowsampler import FlowSampler
import numpy as np
from unittest.mock import MagicMock, create_autospec, patch


@pytest.fixture()
def flow_sampler():
    return create_autospec(FlowSampler)


@pytest.fixture(autouse=True)
def reset_handlers():
    """Reset signal handling after each test"""
    yield
    # Just in case the tests were ran on Windows
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
    except AttributeError:
        print('Cannot set signal attributes on this system.')


@pytest.mark.parametrize('resume', [False, True])
def test_init_no_resume_file(flow_sampler, tmp_path, resume):
    """Test the init method when there is no run to resume from"""

    model = MagicMock()
    output = tmp_path / 'init'
    output.mkdir()
    output = str(output)
    resume = resume
    exit_code = 131
    max_threads = 2
    resume_file = 'test.pkl'
    kwargs = dict(
        nlive=1000,
        pytorch_threads=1,
    )

    flow_sampler.save_kwargs = MagicMock()

    with patch('nessai.flowsampler.NestedSampler', return_value='ns') as mock,\
         patch('nessai.flowsampler.configure_threads') as mock_threads:
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            resume=resume,
            exit_code=exit_code,
            max_threads=max_threads,
            resume_file=resume_file,
            **kwargs,
        )

    mock_threads.assert_called_once_with(
        max_threads=max_threads,
        pytorch_threads=1,
        n_pool=None,
    )

    mock.assert_called_once_with(
        model,
        output=os.path.join(output, ''),
        resume_file=resume_file,
        **kwargs,
    )

    assert flow_sampler.ns == 'ns'

    flow_sampler.save_kwargs.assert_called_once_with(
        kwargs
    )


@pytest.mark.parametrize(
    'test_old, error',
    [(False, None), (True, RuntimeError), (True, FileNotFoundError)]
)
def test_init_resume(flow_sampler, tmp_path, test_old, error):
    """Test the init method when the sampler should resume.

    Tests the case where the first file works and the case where the first
    file fails but the old method works.
    """
    model = MagicMock()
    output = tmp_path / "test"
    output.mkdir()
    resume = True
    exit_code = 131
    max_threads = 2
    resume_file = 'test.pkl'
    weights_file = 'model.pt'
    flow_config = dict(lr=0.1)

    if test_old:
        expected_rf = output / (resume_file + '.old')
        side_effect = [error, 'ns']
    else:
        expected_rf = output / resume_file
        side_effect = ['ns']
    expected_rf.write_text('contents')

    output = str(output)
    expected_rf = str(expected_rf)

    kwargs = dict(
        nlive=1000,
        pytorch_threads=1,
        flow_config=flow_config,
    )

    flow_sampler.save_kwargs = MagicMock()

    with patch('nessai.flowsampler.NestedSampler.resume',
               side_effect=side_effect) as mock_resume,\
         patch('nessai.flowsampler.configure_threads') as mock_threads:
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            resume=resume,
            exit_code=exit_code,
            max_threads=max_threads,
            resume_file=resume_file,
            weights_file=weights_file,
            **kwargs,
        )

    mock_threads.assert_called_once_with(
        max_threads=max_threads,
        pytorch_threads=1,
        n_pool=None,
    )

    mock_resume.assert_called_with(
        expected_rf, model, flow_config=flow_config, weights_file=weights_file,
    )

    assert flow_sampler.ns == 'ns'

    flow_sampler.save_kwargs.assert_called_once_with(
        kwargs
    )


def test_init_resume_error_cannot_resume(flow_sampler, tmp_path):
    """Assert an error is raised if neither file loads"""
    model = MagicMock()
    output = tmp_path / "test"
    output.mkdir()
    resume = True
    resume_file = 'test.pkl'
    expected_rf = output / (resume_file + '.old')
    expected_rf.write_text('contents')
    side_effect = [RuntimeError, RuntimeError]
    output = str(output)
    expected_rf = str(expected_rf)

    assert os.path.exists(expected_rf)

    with patch('nessai.flowsampler.NestedSampler.resume',
               side_effect=side_effect), \
         patch('nessai.flowsampler.configure_threads'), \
         pytest.raises(RuntimeError) as excinfo:
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            resume=resume,
            resume_file=resume_file,
            flow_config=None,
        )
    assert 'Could not resume sampler with error: ' in str(excinfo.value)


def test_init_resume_error_no_file(flow_sampler, tmp_path):
    """Assert an error is raised if resume=True and a resume_file is \
        not specified.
    """
    model = MagicMock()
    output = tmp_path / "test"
    output.mkdir()
    output = str(output)

    with patch('nessai.flowsampler.configure_threads'), \
         pytest.raises(RuntimeError) as excinfo:
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            resume=True,
            resume_file=None,
        )
    assert '`resume_file` must be specified' in str(excinfo.value)


def test_init_signal_handling_enabled(flow_sampler, tmp_path):
    """Assert signal.signal is called when signal handling is enabled."""
    model = MagicMock()
    output = tmp_path / "test"
    output.mkdir()
    output = str(output)

    with patch("signal.signal") as mocked_fn:
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            signal_handling=True
        )
    mocked_fn.assert_called()


def test_init_signal_handling_disabled(flow_sampler, tmp_path):
    """Assert signal handling is not configure when disabled"""
    model = MagicMock()
    output = tmp_path / "test"
    output.mkdir()
    output = str(output)

    with patch("signal.signal") as mocked_fn:
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            signal_handling=False
        )
    mocked_fn.assert_not_called()


def test_init_signal_handling_error(flow_sampler, tmp_path, caplog):
    """Assert signal handling is skipped if an error is raised."""
    model = MagicMock()
    output = tmp_path / "test"
    output.mkdir()
    output = str(output)

    with patch("signal.signal", side_effect=AttributeError):
        FlowSampler.__init__(
            flow_sampler,
            model,
            output=output,
            signal_handling=True
        )
    assert 'Cannot set signal attributes' in str(caplog.text)


@pytest.mark.parametrize('save', [False, True])
@pytest.mark.parametrize('plot', [False, True])
@patch(
    'nessai.flowsampler.draw_posterior_samples', return_value=np.array([0.1])
)
@patch('nessai.plot.plot_live_points')
@patch('nessai.plot.plot_indices')
def test_run(
    mock_plot_indices, mock_plot_post, mock_draw_post, flow_sampler, save, plot
):
    """Test the run method"""
    nlive = 10
    log_Z = -5.0
    nested_samples = [0.1, 1.0, 10.0]
    insertion_indices = [1, 2, 3]
    output = './'
    flow_sampler.ns = MagicMock()
    flow_sampler.ns.nlive = nlive
    flow_sampler.ns.insertion_indices = insertion_indices
    flow_sampler.output = output
    flow_sampler.ns.initialise = MagicMock()
    flow_sampler.ns.nested_sampling_loop = MagicMock(
        return_value=[log_Z, nested_samples]
    )
    flow_sampler.ns.state = MagicMock()
    flow_sampler.ns.state.plot_state = MagicMock()
    flow_sampler.save_results = MagicMock()

    FlowSampler.run(flow_sampler, save=save, plot=plot)

    flow_sampler.ns.initialise.assert_called_once()
    mock_draw_post.assert_called_once_with(nested_samples, nlive)
    if save:
        flow_sampler.save_results.assert_called_once()
    else:
        flow_sampler.save_results.assert_not_called()

    if plot:
        mock_plot_indices.assert_called_once_with(
           insertion_indices,
           nlive,
           filename=os.path.join(output, 'insertion_indices.png'),
        )
        mock_plot_post.assert_called_once_with(
            flow_sampler.posterior_samples,
            filename=os.path.join(output, 'posterior_distribution.png'),
        )
        flow_sampler.ns.state.plot.assert_called_once_with(
            os.path.join(output, 'logXlogL.png'),
        )
    else:
        mock_plot_indices.assert_not_called()
        mock_plot_post.assert_not_called()
        flow_sampler.ns.state.plot.assert_not_called()

    assert flow_sampler.logZ == log_Z
    assert flow_sampler.nested_samples == nested_samples
    np.testing.assert_array_equal(
        flow_sampler.posterior_samples, np.array([0.1])
    )


@pytest.mark.parametrize('test_class', [False, True])
def test_save_kwargs(flow_sampler, tmpdir, test_class):
    """Test the save kwargs method.

    If `test_class` is true, tests the case of the flow class being a class
    rather than a string.
    """
    kwargs = dict(nlive=10, a=np.array([0.1]))
    if test_class:
        from nessai.proposal import FlowProposal
        kwargs['flow_class'] = FlowProposal
    else:
        kwargs['flow_class'] = 'flowproposal'

    flow_sampler.output = str(tmpdir.mkdir('test'))

    FlowSampler.save_kwargs(flow_sampler, kwargs)

    assert os.path.exists(os.path.join(flow_sampler.output, 'config.json'))


def test_save_results(flow_sampler, tmpdir):
    """Test the save results method"""
    from datetime import timedelta
    output = str(tmpdir.mkdir('test'))
    filename = os.path.join(output, 'result.json')

    ns = MagicMock()
    ns.nlive = 1
    ns.iteration = 3
    ns.min_likelihood = [-3, -2, 1]
    ns.max_likelihood = [1, 2, 3]
    ns.likelihood_evaluations = 3
    ns.logZ_history = [1, 2, 3]
    ns.mean_acceptance_history = [1, 2, 3]
    ns.rolling_p = [0.5]
    ns.population_iterations = []
    ns.population_acceptance = []
    ns.training_iterations = []
    ns.insertion_indices = []
    ns.sampling_time = timedelta()
    ns.training_time = timedelta()
    ns.proposal_population_time = timedelta()
    ns.likelihood_evaluation_time = timedelta(2)

    flow_sampler.ns = ns
    flow_sampler.posterior_samples = np.array([0.1], dtype=[('x', 'f8')])
    flow_sampler.nested_samples = np.array([0.1], dtype=[('x', 'f8')])

    FlowSampler.save_results(flow_sampler, filename)

    assert os.path.exists(filename)


def test_safe_exit(flow_sampler):
    """Test the safe exit method."""
    flow_sampler.exit_code = 130
    flow_sampler.ns = MagicMock()
    flow_sampler.ns.checkpoint = MagicMock()
    flow_sampler.ns.model = MagicMock()
    flow_sampler.ns.model.close_pool = MagicMock()

    with patch('sys.exit') as mock_exit:
        FlowSampler.safe_exit(flow_sampler, signum=2)

    mock_exit.assert_called_once_with(130)
    flow_sampler.ns.checkpoint.assert_called_once()
    flow_sampler.ns.model.close_pool.assert_called_once_with(code=2)


@pytest.mark.parametrize(
    "kwargs",
    [dict(n_pool=None), dict(max_threads=3, n_pool=2)]
)
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_signal_handling(tmp_path, caplog, model, kwargs):
    """Test the signal handling in nessai.

    Test is based on a similar test in bilby which is in turn based on: \
        https://stackoverflow.com/a/49615525/18400311
    """
    output = tmp_path / "output"
    output.mkdir()

    fs = FlowSampler(
        model,
        output=output,
        nlive=500,
        poolsize=1000,
        exit_code=5,
        signal_handling=True,
        **kwargs
    )

    pid = os.getpid()

    def trigger_signal():
        time.sleep(4)
        os.kill(pid, signal.SIGINT)

    thread = Thread(target=trigger_signal)
    thread.daemon = True
    thread.start()

    with pytest.raises(SystemExit):
        try:
            while True:
                fs.run(save=False, plot=False)
        except SystemExit as error:
            assert error.code == 5
            raise

    assert f"Trying to safely exit with code {signal.SIGINT}" \
        in str(caplog.text)


@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_signal_handling_disabled(tmp_path, caplog, model):
    """Assert signal handling is correctly disabled.

    Test is based on a similar test in bilby which is in turn based on: \
        https://stackoverflow.com/a/49615525/18400311
    """
    output = tmp_path / "output"
    output.mkdir()

    fs = FlowSampler(
        model,
        output=output,
        nlive=500,
        poolsize=1000,
        exit_code=4,
        signal_handling=False,
    )

    pid = os.getpid()

    def trigger_signal():
        time.sleep(4)
        os.kill(pid, signal.SIGINT)

    thread = Thread(target=trigger_signal)
    thread.daemon = True
    thread.start()

    # Need to catch SIGINT to prevent tests for exiting
    def handler(signum=None, frame=None):
        sys.exit(10)

    signal.signal(signal.SIGINT, handler)

    with pytest.raises(SystemExit):
        try:
            while True:
                fs.run(save=False, plot=False)
        except SystemExit as error:
            assert error.code == 10
            raise

    assert "Signal handling is disabled" in str(caplog.text)
