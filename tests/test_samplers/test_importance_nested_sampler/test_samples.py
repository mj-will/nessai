"""Tests related to how samples are handled"""
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
