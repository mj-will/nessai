"""Tests related to how samples are handled"""
from unittest.mock import MagicMock

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
from nessai.utils.testing import assert_structured_arrays_equal
import numpy as np
import pytest


def test_sort_points(ins, samples):
    order = np.argsort(samples["logL"])
    extra = np.arange(samples.size)
    sorted_samples, sorted_extra = INS.sort_points(ins, samples, extra)
    assert_structured_arrays_equal(sorted_samples, samples[order])
    np.testing.assert_array_equal(sorted_extra, extra[order])


def test_populate_live_points(ins, model):
    n = 100
    ins.n_initial = n
    ins.model = model

    def fn(x):
        return np.sort(x, order="logL")

    ins.sort_points = MagicMock(side_effect=fn)

    INS.populate_live_points(ins)

    assert len(ins.samples) == n
    np.testing.assert_array_equal(ins.live_points_indices, np.arange(n))
    assert np.isfinite(ins.samples["logL"]).all()
    assert np.isfinite(ins.samples["logW"]).all()
    assert np.isfinite(ins.samples["logQ"]).all()
    assert np.isfinite(ins.samples["logP"]).all()
    assert (ins.samples["it"] == -1).all()


def test_add_to_nested_samples(ins):
    ns_indices = np.array([0, 1, 2, 4, 5, 8])
    indices = np.array([3, 6, 7, 9])
    ins.nested_samples_indices = ns_indices

    INS.add_to_nested_samples(ins, indices)

    np.testing.assert_array_equal(ins.nested_samples_indices, np.arange(10))


@pytest.mark.parametrize("has_live_points", [True, False])
def test_add_samples(ins, samples, log_q, has_live_points):
    n = int(0.8 * samples.size)

    if has_live_points:
        n_ns = int(0.8 * n)
        ns_indices = np.sort(np.random.choice(n, size=n_ns, replace=False))
        live_indices = np.sort(list(set(np.arange(n)) - set(ns_indices)))
    else:
        n_ns = n
        ns_indices = np.arange(n_ns)
        live_indices = None

    sort_idx = np.argsort(samples[:n], order="logL")
    ins.samples = samples[:n][sort_idx]
    ins.log_q = log_q[:n][sort_idx]
    ins.nested_samples_indices = ns_indices
    ins.live_points_indices = live_indices

    sort_idx = np.argsort(samples[n:], order="logL")
    new_samples = samples[n:][sort_idx]
    new_log_q = log_q[n:][sort_idx]

    INS.add_samples(ins, new_samples, new_log_q)

    assert len(ins.live_points_indices) == (n - n_ns + new_samples.size)

    assert np.all(np.diff(ins.samples["logL"]) >= 0)
    assert np.all(np.diff(ins.samples[ins.live_points_indices]["logL"]) >= 0)
    assert np.all(
        np.diff(ins.samples[ins.nested_samples_indices]["logL"]) >= 0
    )


@pytest.mark.parametrize("n", [5, 10])
def test_remove_samples(ins, n):
    ins.history = dict(n_removed=[5])
    ins.replace_all = False
    ins.add_to_nested_samples = MagicMock()

    live_points_indices = np.random.choice(20, size=15, replace=False)
    ins.live_points_indices = live_points_indices.copy()

    INS.remove_samples(ins, n)

    assert ins.history["n_removed"] == [5, n]
    np.testing.assert_array_equal(
        ins.add_to_nested_samples.call_args[0][0], live_points_indices[:n]
    )
    np.testing.assert_array_equal(
        ins.live_points_indices, live_points_indices[n:]
    )


def test_remove_samples_replace_all(ins):
    ins.history = dict(n_removed=[5])
    ins.replace_all = True
    ins.add_to_nested_samples = MagicMock()

    live_points_indices = np.random.choice(20, size=15, replace=False)
    ins.live_points_indices = live_points_indices.copy()
    ins.live_points = np.ones(live_points_indices.size)

    INS.remove_samples(ins, 10)

    assert ins.history["n_removed"] == [5, live_points_indices.size]
    np.testing.assert_array_equal(
        ins.add_to_nested_samples.call_args[0][0], live_points_indices
    )
    assert ins.live_points_indices is None
