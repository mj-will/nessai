"""Tests for level-related methods"""
from unittest.mock import MagicMock

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
import numpy as np
import pytest


@pytest.mark.parametrize("include_likelihood", [False, True])
@pytest.mark.parametrize("use_log_weights", [False, True])
def test_determine_level_entropy(
    ins, samples, include_likelihood, use_log_weights
):
    ins.live_points = np.sort(samples, order="logL")
    ins.plot = False
    n = INS.determine_level_entropy(
        ins,
        q=0.5,
        use_log_weights=use_log_weights,
        include_likelihood=include_likelihood,
    )
    assert 0 < n < samples.size


@pytest.mark.parametrize("include_likelihood", [False, True])
def test_determine_level_quantile(ins, samples, include_likelihood):
    ins.live_points = np.sort(samples, order="logL")
    ins.plot = False
    n = INS.determine_level_quantile(
        ins,
        q=0.8,
        include_likelihood=include_likelihood,
    )
    assert 0 < n < samples.size


def test_determine_level_select_quantile(ins):
    n = 10
    ins.determine_level_quantile = MagicMock(return_value=n)
    out = INS.determine_level(ins, "quantile", q=0.8)
    ins.determine_level_quantile.assert_called_once_with(q=0.8)
    assert out == n


def test_determine_level_select_entropy(ins):
    n = 10
    ins.determine_level_entropy = MagicMock(return_value=n)
    out = INS.determine_level(ins, "entropy", q=0.8)
    ins.determine_level_entropy.assert_called_once_with(q=0.8)
    assert out == n
