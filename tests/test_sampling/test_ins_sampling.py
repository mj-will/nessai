"""Test sampling with the importance nested sampler"""
import os

from nessai.flowsampler import FlowSampler
import numpy as np
import pytest


@pytest.mark.slow_integration_test
@pytest.mark.flaky(reruns=3)
def test_ins_resume(tmp_path, model, flow_config):
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
