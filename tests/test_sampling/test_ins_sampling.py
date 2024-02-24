"""Test sampling with the importance nested sampler"""

import os
import pickle

from nessai.flowsampler import FlowSampler
import numpy as np
import pytest


@pytest.mark.slow_integration_test
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("save_log_q", [False, True])
def test_ins_resume(tmp_path, model, flow_config, save_log_q):
    """Assert the INS sampler resumes correctly"""
    output = tmp_path / "test_ins_resume"
    fp = FlowSampler(
        model,
        output=output,
        resume=True,
        nlive=500,
        min_samples=50,
        plot=False,
        flow_config=flow_config,
        importance_nested_sampler=True,
        max_iteration=2,
        save_log_q=save_log_q,
    )
    fp.run()

    original_log_q = fp.ns.log_q.copy()

    assert fp.ns.iteration == 2
    assert os.path.exists(os.path.join(output, "nested_sampler_resume.pkl"))

    fp = FlowSampler(
        model,
        output=output,
        resume=True,
        flow_config=flow_config,
        plot=False,
        importance_nested_sampler=True,
    )
    new_log_q = fp.ns.log_q.copy()

    assert fp.ns.max_iteration == 2
    assert fp.ns.finalised is True
    np.testing.assert_array_almost_equal(new_log_q, original_log_q)


@pytest.mark.slow_integration_test
def test_ins_checkpoint_callback(tmp_path, model, flow_config):
    output = tmp_path / "test_ins_checkpoint_callback"

    filename = os.path.join(output, "test.pkl")
    resume_file = "resume.pkl"

    def checkpoint_callback(state):
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    fs = FlowSampler(
        model,
        output=output,
        resume=True,
        nlive=500,
        min_samples=50,
        plot=False,
        flow_config=flow_config,
        checkpoint_on_iteration=True,
        checkpoint_interval=1,
        importance_nested_sampler=True,
        max_iteration=2,
        resume_file=resume_file,
        checkpoint_callback=checkpoint_callback,
    )
    fs.run()
    assert fs.ns.iteration == 2
    assert os.path.exists(filename)
    assert not os.path.exists(os.path.join(output, resume_file))

    del fs

    with open(filename, "rb") as f:
        resume_data = pickle.load(f)

    resume_data.test_variable = "abc"

    fs = FlowSampler(
        model,
        output=output,
        resume=True,
        nlive=500,
        min_samples=50,
        plot=False,
        flow_config=flow_config,
        checkpoint_on_iteration=True,
        checkpoint_interval=1,
        importance_nested_sampler=True,
        max_iteration=2,
        checkpoint_callback=checkpoint_callback,
        resume_data=resume_data,
        resume_file=resume_file,
    )
    assert fs.ns.iteration == 2
    assert fs.ns.finalised is True
    assert fs.ns.test_variable == "abc"
