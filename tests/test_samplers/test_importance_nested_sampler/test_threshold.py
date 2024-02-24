"""Tests for level-related methods"""

import os
from unittest.mock import MagicMock

from nessai.samplers.importancesampler import (
    ImportanceNestedSampler as INS,
    OrderedSamples,
)
import numpy as np
import pytest


@pytest.mark.parametrize("include_likelihood", [False, True])
@pytest.mark.parametrize("use_log_weights", [False, True])
def test_determine_threshold_entropy(
    ins, samples, include_likelihood, use_log_weights
):
    samples = np.sort(samples, order="logL")
    ins.plot = False
    n = INS.determine_threshold_entropy(
        ins,
        samples,
        q=0.5,
        use_log_weights=use_log_weights,
        include_likelihood=include_likelihood,
    )
    assert 0 < n < samples.size


def test_determine_threshold_entropy_plot(ins, samples, tmp_path):
    samples = np.sort(samples, order="logL")
    ins.plot = True
    ins._plot_level_cdf = True
    ins.output = tmp_path / "test_entropy_plot"
    ins.plot_level_cdf = MagicMock()
    ins.iteration = 2
    n = INS.determine_threshold_entropy(
        ins,
        samples,
        q=0.5,
    )
    assert 0 < n < samples.size
    ins.plot_level_cdf.assert_called_once()
    assert os.path.exists(os.path.join(ins.output, "levels", "level_2"))


@pytest.mark.parametrize("include_likelihood", [False, True])
def test_determine_threshold_quantile(ins, samples, include_likelihood):
    samples = np.sort(samples, order="logL")
    ins.plot = False
    n = INS.determine_threshold_quantile(
        ins,
        samples,
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
    ins.max_samples = None
    ins.draw_constant = True
    ins.nlive = n_live
    samples = np.empty(n_live, dtype=[("x", "f8"), ("logL", "f8")])
    samples["logL"] = 10 * np.arange(n_live)

    expected_log_l = samples["logL"][expected]

    ins.determine_threshold_quantile = MagicMock(return_value=n)
    ins.determine_threshold_entropy = MagicMock(return_value=n)

    out = INS.determine_log_likelihood_threshold(
        ins, samples, method=method, q=0.8
    )

    if method == "entropy":
        ins.determine_threshold_entropy.assert_called_once_with(samples, q=0.8)
        ins.determine_threshold_quantile.assert_not_called()
    else:
        ins.determine_threshold_quantile.assert_called_once_with(
            samples, q=0.8
        )
        ins.determine_threshold_entropy.assert_not_called()

    assert out == expected_log_l


@pytest.mark.parametrize(
    "n_samples, n_remove, min_remove, "
    "min_samples, max_samples, n_live, expected",
    [
        [50, 10, 5, 10, 55, 30, 25],
        [56, 10, 5, 10, 55, 30, 31],
        [50, 20, 5, 10, 100, 30, 20],
        [1601, 100, 50, 50, 1600, 200, 201],
    ],
)
def test_determine_threshold_max_samples(
    ins,
    n_samples,
    n_remove,
    min_remove,
    min_samples,
    max_samples,
    n_live,
    expected,
    caplog,
):
    ins.min_samples = min_samples
    ins.min_remove = min_remove
    ins.max_samples = max_samples
    ins.draw_constant = True
    ins.nlive = n_live
    samples = np.empty(n_samples, dtype=[("x", "f8"), ("logL", "f8")])
    samples["logL"] = 10 * np.arange(n_samples)

    expected_log_l = samples["logL"][expected]

    ins.determine_threshold_entropy = MagicMock(return_value=n_remove)

    out = INS.determine_log_likelihood_threshold(
        ins, samples, method="entropy", q=0.8
    )

    ins.determine_threshold_entropy.assert_called_once_with(samples, q=0.8)
    assert out == expected_log_l

    if expected != n_remove:
        assert "Next level would have more than max samples" in str(
            caplog.text
        )


def test_update_log_likelihood_threshold(ins, iid):
    threshold = 10
    ins.training_samples = MagicMock(spec=OrderedSamples)
    if iid:
        ins.iid_samples = MagicMock(spec=OrderedSamples)
    INS.update_log_likelihood_threshold(ins, threshold)

    ins.training_samples.update_log_likelihood_threshold.assert_called_once_with(  # noqa
        threshold
    )
    if iid:
        ins.iid_samples.update_log_likelihood_threshold.assert_called_once_with(  # noqa
            threshold
        )
