"""Tests for level-related methods"""
import os
from unittest.mock import MagicMock

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
import numpy as np
import pytest


@pytest.mark.parametrize("include_likelihood", [False, True])
@pytest.mark.parametrize("use_log_weights", [False, True])
def test_determine_threshold_entropy(
    ins, samples, include_likelihood, use_log_weights
):
    ins.live_points = np.sort(samples, order="logL")
    ins.plot = False
    n = INS.determine_threshold_entropy(
        ins,
        q=0.5,
        use_log_weights=use_log_weights,
        include_likelihood=include_likelihood,
    )
    assert 0 < n < samples.size


def test_determine_threshold_entropy_plot(ins, samples, tmp_path):
    ins.live_points = np.sort(samples, order="logL")
    ins.plot = True
    ins._plot_level_cdf = True
    ins.output = tmp_path / "test_entropy_plot"
    ins.plot_level_cdf = MagicMock()
    ins.iteration = 2
    n = INS.determine_threshold_entropy(
        ins,
        q=0.5,
    )
    assert 0 < n < samples.size
    ins.plot_level_cdf.assert_called_once()
    assert os.path.exists(os.path.join(ins.output, "levels", "level_2"))


@pytest.mark.parametrize("include_likelihood", [False, True])
def test_determine_threshold_quantile(ins, samples, include_likelihood):
    ins.live_points = np.sort(samples, order="logL")
    ins.plot = False
    n = INS.determine_threshold_quantile(
        ins,
        q=0.8,
        include_likelihood=include_likelihood,
    )
    assert 0 < n < samples.size


@pytest.mark.parametrize(
    "n, min_remove, min_samples, n_live, expected",
    [
        [10, 5, 8, 20, 10],
        [4, 5, 8, 20, 5],
        [15, 5, 10, 20, 10],
    ],
)
@pytest.mark.parametrize("method", ["entropy", "quantile"])
def test_determine_threshold(
    ins,
    n,
    min_remove,
    min_samples,
    n_live,
    expected,
    method,
):
    ins.min_samples = min_samples
    ins.min_remove = min_remove
    ins.live_points = np.empty(n_live, dtype=[("x", "f8"), ("logL", "f8")])
    ins.live_points["logL"] = np.arange(n_live)

    ins.determine_threshold_quantile = MagicMock(return_value=n)
    ins.determine_threshold_entropy = MagicMock(return_value=n)

    out = INS.determine_likelihood_threshold(ins, method, q=0.8)

    if method == "entropy":
        ins.determine_threshold_entropy.assert_called_once_with(q=0.8)
        ins.determine_threshold_quantile.assert_not_called()
    else:
        ins.determine_threshold_quantile.assert_called_once_with(q=0.8)
        ins.determine_threshold_entropy.assert_not_called()

    assert out == expected
