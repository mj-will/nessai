"""Tests for level-related methods"""
from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
import pytest


@pytest.mark.parametrize("include_likelihood", [False, True])
@pytest.mark.parametrize("use_log_weights", [False, True])
def test_determine_level_entropy(
    ins, samples, include_likelihood, use_log_weights
):
    ins.live_points = samples
    ins.plot = False
    n = INS.determine_level_entropy(
        ins,
        q=0.5,
        use_log_weights=use_log_weights,
        include_likelihood=include_likelihood,
    )
    assert 0 < n < samples.size
