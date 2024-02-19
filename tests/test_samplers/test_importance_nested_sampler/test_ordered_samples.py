from nessai.samplers.importancesampler import OrderedSamples
from nessai.evidence import _INSIntegralState
from nessai.utils.testing import assert_structured_arrays_equal
import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec, patch


@pytest.fixture()
def ordered_samples():
    return create_autospec(OrderedSamples)


@pytest.fixture()
def samples(samples):
    return np.sort(samples, order="logL")


@pytest.fixture()
def live_points(samples):
    return samples[samples.size // 2 :]


@pytest.fixture()
def nested_samples(samples):
    return samples[: samples.size // 2]


def test_live_points(ordered_samples, samples):
    indices = [2, 3]
    ordered_samples.live_points_indices = indices
    ordered_samples.samples = samples
    out = OrderedSamples.live_points.__get__(ordered_samples)
    assert_structured_arrays_equal(out, samples[indices])


def test_live_points_none(ordered_samples, samples):
    ordered_samples.live_points_indices = None
    ordered_samples.samples = samples
    assert OrderedSamples.live_points.__get__(ordered_samples) is None


def test_live_points_setter_error(ordered_samples):
    with pytest.raises(ValueError, match=r"Can only set live points to None"):
        OrderedSamples.live_points.__set__(ordered_samples, 1.0)


def test_live_points_setter(ordered_samples):
    OrderedSamples.live_points.__set__(ordered_samples, None)
    assert ordered_samples.live_points_indices is None


def test_nested_samples(ordered_samples, samples):
    indices = [2, 3]
    ordered_samples.nested_samples_indices = indices
    ordered_samples.samples = samples
    out = OrderedSamples.nested_samples.__get__(ordered_samples)
    assert_structured_arrays_equal(out, samples[indices])


def test_nested_samples_none(ordered_samples, samples):
    ordered_samples.nested_samples_indices = None
    ordered_samples.samples = samples
    assert OrderedSamples.nested_samples.__get__(ordered_samples) is None


def test_update_log_likelihood_threshold(ordered_samples):
    val = 5.0
    ordered_samples.log_likelihood_threshold = None
    OrderedSamples.update_log_likelihood_threshold(ordered_samples, val)
    assert ordered_samples.log_likelihood_threshold == val


def test_sort_samples_only(ordered_samples):
    samples = np.array(np.random.randn(10), [("logL", "f8")])
    samples_out = OrderedSamples.sort_samples(ordered_samples, samples)
    assert np.all(np.diff(samples_out["logL"]) > 0)


def test_sort_samples_with_log_q(ordered_samples):
    samples = np.array(np.random.randn(10), [("logL", "f8")])
    order = np.argsort(samples["logL"])
    extra = np.arange(samples.size)
    sorted_samples, sorted_extra = OrderedSamples.sort_samples(
        ordered_samples, samples, extra
    )
    assert_structured_arrays_equal(sorted_samples, samples[order])
    np.testing.assert_array_equal(sorted_extra, extra[order])


def test_add_initial_samples(ordered_samples, samples, log_q):
    ordered_samples.sort_samples = MagicMock(return_value=(samples, log_q))
    OrderedSamples.add_initial_samples(ordered_samples, samples, log_q)
    ordered_samples.sort_samples.assert_called_once_with(samples, log_q)
    assert ordered_samples.samples is samples
    assert ordered_samples.log_q is log_q
    assert len(ordered_samples.live_points_indices) == len(samples)
    assert np.all(np.diff(ordered_samples.live_points_indices) > 0)
    np.testing.assert_array_equal(
        ordered_samples.live_points_indices, np.arange(samples.size)
    )


@pytest.mark.parametrize("has_live_points", [True, False])
def test_add_samples_soft(ordered_samples, samples, log_q, has_live_points):
    n = int(0.8 * samples.size)
    ordered_samples.strict_threshold = False

    if has_live_points:
        n_ns = int(0.8 * n)
        ns_indices = np.sort(np.random.choice(n, size=n_ns, replace=False))
        live_indices = np.sort(list(set(np.arange(n)) - set(ns_indices)))
    else:
        n_ns = n
        ns_indices = np.arange(n_ns)
        live_indices = None

    sort_idx = np.argsort(samples[:n], order="logL")
    ordered_samples.samples = samples[:n][sort_idx]
    ordered_samples.log_q = log_q[:n][sort_idx]
    ordered_samples.nested_samples_indices = ns_indices
    ordered_samples.live_points_indices = live_indices

    new_samples = samples[n:]
    new_log_q = log_q[n:]

    idx = np.argsort(new_samples, order="logL")
    new_samples_ordered = new_samples[idx]
    new_log_q_ordered = new_log_q[idx]

    ordered_samples.sort_samples = MagicMock(
        return_value=(new_samples_ordered, new_log_q_ordered)
    )

    OrderedSamples.add_samples(ordered_samples, new_samples, new_log_q)

    ordered_samples.sort_samples.assert_called_once_with(
        new_samples, new_log_q
    )
    assert len(ordered_samples.live_points_indices) == (
        n - n_ns + new_samples.size
    )
    assert np.all(np.diff(ordered_samples.samples["logL"]) >= 0)
    assert np.all(
        np.diff(
            ordered_samples.samples[ordered_samples.live_points_indices][
                "logL"
            ]
        )
        >= 0
    )
    assert np.all(
        np.diff(
            ordered_samples.samples[ordered_samples.nested_samples_indices][
                "logL"
            ]
        )
        >= 0
    )


def test_add_samples(ordered_samples, samples, log_q):
    ordered_samples.strict_threshold = True

    idx = np.argsort(samples, order="logL")
    expected = samples[idx].copy()
    expected_log_q = log_q[idx].copy()

    perm = np.random.permutation(samples.size)
    samples = samples[perm]
    log_q = log_q[perm]
    n = int(0.8 * samples.size)
    idx = np.argsort(samples[:n], order="logL")
    ordered_samples.samples = samples[:n][idx]
    ordered_samples.log_q = log_q[:n][idx]

    new_samples = samples[n:]
    new_log_q = log_q[n:]

    idx = np.argsort(new_samples, order="logL")
    new_samples_ordered = new_samples[idx]
    new_log_q_ordered = new_log_q[idx]
    ordered_samples.sort_samples = MagicMock(
        return_value=(new_samples_ordered, new_log_q_ordered)
    )

    ordered_samples.log_likelihood_threshold = new_samples_ordered[
        new_samples_ordered.size // 2
    ]["logL"].item()

    n_expected = np.sum(
        expected["logL"] >= ordered_samples.log_likelihood_threshold
    )

    OrderedSamples.add_samples(ordered_samples, new_samples, new_log_q)

    assert_structured_arrays_equal(ordered_samples.samples, expected)
    np.testing.assert_array_equal(ordered_samples.log_q, expected_log_q)
    np.testing.assert_array_equal(
        ordered_samples.nested_samples_indices,
        np.arange(samples.size - n_expected),
    )
    np.testing.assert_array_equal(
        ordered_samples.live_points_indices,
        np.arange(samples.size - n_expected, samples.size),
    )


@pytest.mark.parametrize("replace_all", [False, True])
def test_remove_samples(ordered_samples, replace_all):
    n = 10
    ordered_samples.live_points = np.array(
        np.arange(n), dtype=[("logL", "f8")]
    )
    ordered_samples.live_points_indices = np.arange(n)
    ordered_samples.log_likelihood_threshold = 5.5
    ordered_samples.replace_all = replace_all
    ordered_samples.add_to_nested_samples = MagicMock()

    expected = n if replace_all else 6

    out = OrderedSamples.remove_samples(ordered_samples)

    assert out == expected
    ordered_samples.add_to_nested_samples.assert_called_once()
    np.testing.assert_array_equal(
        ordered_samples.add_to_nested_samples.call_args.args[0],
        np.arange(expected),
    )
    if replace_all:
        assert ordered_samples.live_points_indices is None
    else:
        np.testing.assert_array_equal(
            ordered_samples.live_points_indices, np.arange(6, n)
        )


def test_add_to_nested_samples(ordered_samples):
    ns_indices = np.array([0, 1, 2, 4, 5, 8])
    indices = np.array([3, 6, 7, 9])
    ordered_samples.nested_samples_indices = ns_indices
    OrderedSamples.add_to_nested_samples(ordered_samples, indices)
    np.testing.assert_array_equal(
        ordered_samples.nested_samples_indices, np.arange(10)
    )


def test_update_evidence(ordered_samples, live_points, nested_samples):
    ordered_samples.state = create_autospec(_INSIntegralState)
    ordered_samples.live_points = live_points
    ordered_samples.nested_samples = nested_samples
    OrderedSamples.update_evidence(ordered_samples)
    ordered_samples.state.update_evidence.assert_called_once_with(
        nested_samples=nested_samples,
        live_points=live_points,
    )


def test_finalise(ordered_samples, samples):
    ordered_samples.samples = samples
    ordered_samples.live_points_indices = np.arange(4)

    def add_to_ns(indices):
        ordered_samples.live_points_indices = None

    ordered_samples.add_to_nested_samples = MagicMock(side_effect=add_to_ns)
    ordered_samples.state = create_autospec(_INSIntegralState)

    OrderedSamples.finalise(ordered_samples)

    ordered_samples.state.update_evidence.assert_called_once_with(samples)
    assert ordered_samples.live_points is None
    assert ordered_samples.live_points_indices is None


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_compute_importance(ordered_samples, log_q, samples, ratio):
    ordered_samples.samples = samples
    ordered_samples.log_q = log_q
    out = OrderedSamples.compute_importance(
        ordered_samples, importance_ratio=ratio
    )
    assert len(set(out.keys()) - {"total", "posterior", "evidence"}) == 0
    assert np.all(np.isfinite(list(out.values())))


@pytest.mark.parametrize("threshold", [None, 0])
def test_computed_evidence_ratio(ordered_samples, samples, threshold):
    log_z_total = -10.0
    log_z = -6.0
    ordered_samples.samples = samples
    ordered_samples.log_likelihood_threshold = np.median(samples["logL"])
    ordered_samples.state = MagicMock(spec=_INSIntegralState)
    ordered_samples.state.log_evidence = log_z_total

    with patch(
        "nessai.samplers.importancesampler.log_evidence_from_ins_samples",
        return_value=log_z,
    ) as mock_log_evidence:
        out = OrderedSamples.compute_evidence_ratio(ordered_samples, threshold)
    actual_threshold = (
        ordered_samples.log_likelihood_threshold
        if threshold is None
        else threshold
    )
    above_threshold = samples["logL"] > actual_threshold

    mock_log_evidence.assert_called_once()
    assert_structured_arrays_equal(
        mock_log_evidence.call_args_list[0][0][0], samples[above_threshold]
    )
    assert out == (log_z - log_z_total)


@pytest.mark.parametrize("save_log_q", [False, True])
def test_getstate(ordered_samples, save_log_q):
    samples = np.random.randn(20, 4)
    log_q = np.random.randn(2, 20)
    ordered_samples.save_log_q = save_log_q
    ordered_samples.log_q = log_q
    ordered_samples.samples = samples
    state = OrderedSamples.__getstate__(ordered_samples)
    assert state["samples"] is samples
    if save_log_q:
        assert state["log_q"] is log_q
    else:
        assert state["log_q"] is None
