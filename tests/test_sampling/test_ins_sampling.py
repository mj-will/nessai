"""Test sampling with the importance nested sampler"""
import os

from nessai.flowsampler import FlowSampler
import pytest


@pytest.mark.slow_integration_test
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

    assert fp.ns.max_iteration == 2
    assert fp.ns.finalised is True
