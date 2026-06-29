# -*- coding: utf-8 -*-
"""Standalone tests for truncation rules and helpers."""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from nessai.proposal.flowproposal.truncation import (
    DEFAULT_TRUNCATION_KWARGS,
    DEFAULT_TRUNCATION_METHODS,
    BaseTruncationRule,
    LatentRadiusTruncation,
    LikelihoodThresholdTruncation,
    MinLogQTruncation,
    TruncationScheme,
    apply_default_truncation_config,
    build_truncation_methods,
    get_deprecated_latent_radius_arguments,
    get_deprecated_latent_radius_kwargs,
    get_truncation_rule_class,
    normalise_truncation_kwargs,
    normalise_truncation_methods,
)


@pytest.fixture
def point():
    def _point(x, y, logl=0.0):
        out = np.zeros(1, dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")])
        out["x"] = x
        out["y"] = y
        out["logL"] = logl
        return out[0]

    return _point


@pytest.fixture
def samples():
    def _samples(values):
        out = np.zeros(
            len(values), dtype=[("x", "f8"), ("y", "f8"), ("logL", "f8")]
        )
        out["x"] = [v[0] for v in values]
        out["y"] = [v[1] for v in values]
        return out

    return _samples


class DummyTruncationRule(BaseTruncationRule):
    name = "dummy"
    _transient_defaults = {"_value": np.nan}

    def prepare(self, proposal, worst_point, radius=None):
        self._value = 1.0


def test_get_deprecated_latent_radius_arguments():
    out = get_deprecated_latent_radius_arguments(
        constant_volume_mode=True,
        volume_fraction=None,
        fuzz=1.2,
        expansion_fraction=None,
        fixed_radius=None,
        radius_mode=None,
        min_radius=None,
        max_radius=5.0,
        compute_radius_with_all=None,
    )
    assert out == ["constant_volume_mode", "fuzz", "max_radius"]


def test_get_deprecated_latent_radius_kwargs():
    out = get_deprecated_latent_radius_kwargs(
        constant_volume_mode=True,
        volume_fraction=None,
        fuzz=1.2,
        expansion_fraction=None,
        fixed_radius=None,
        radius_mode=None,
        min_radius=None,
        max_radius=5.0,
        compute_radius_with_all=None,
    )
    assert out == {
        "constant_volume_mode": True,
        "fuzz": 1.2,
        "max_radius": 5.0,
    }


@pytest.mark.parametrize(
    "truncation_method,truncation_methods,expected",
    [
        (None, None, []),
        ("min_log_q", None, ["min_log_q"]),
        (None, "likelihood_threshold", ["likelihood_threshold"]),
        (
            None,
            ["min_log_q", "min_log_q", "likelihood_threshold"],
            ["min_log_q", "likelihood_threshold"],
        ),
    ],
)
def test_normalise_truncation_methods(
    truncation_method, truncation_methods, expected
):
    assert (
        normalise_truncation_methods(truncation_method, truncation_methods)
        == expected
    )


def test_build_truncation_methods_rejects_both_method_and_methods():
    with pytest.raises(
        ValueError, match="Specify only one of truncation_method"
    ):
        build_truncation_methods(
            truncation_method="min_log_q",
            truncation_methods=["likelihood_threshold"],
        )


def test_build_truncation_methods_includes_default_latent_radius():
    assert build_truncation_methods(default_latent_radius=True) == [
        "latent_radius"
    ]


def test_build_truncation_methods_includes_legacy_radius_and_flags():
    assert build_truncation_methods(
        truncate_log_q=True,
        enforce_likelihood_threshold=True,
        latent_radius_kwargs={"fixed_radius": 2.0},
    ) == [
        "latent_radius",
        "min_log_q",
        "likelihood_threshold",
    ]


def test_apply_default_truncation_config():
    methods, kwargs = apply_default_truncation_config(
        ["latent_radius", "min_log_q"],
        truncation_kwargs={"latent_radius": {"fuzz": 1.5}},
    )
    assert methods == ["latent_radius", "min_log_q"]
    assert kwargs["latent_radius"] == {
        **DEFAULT_TRUNCATION_KWARGS["latent_radius"],
        "fuzz": 1.5,
    }


def test_apply_default_truncation_config_with_default_methods():
    methods, kwargs = apply_default_truncation_config(
        [],
        truncation_kwargs=None,
        default_latent_radius=True,
    )
    assert methods == DEFAULT_TRUNCATION_METHODS
    assert kwargs == DEFAULT_TRUNCATION_KWARGS


def test_normalise_truncation_kwargs_none():
    assert normalise_truncation_kwargs() == {}


def test_normalise_truncation_kwargs_wraps_flat_kwargs_for_single_method():
    kwargs = normalise_truncation_kwargs(
        truncation_method="latent_radius",
        truncation_kwargs={"constant_volume_mode": True},
    )
    assert kwargs == {"latent_radius": {"constant_volume_mode": True}}


def test_normalise_truncation_kwargs_keeps_nested_kwargs():
    kwargs = normalise_truncation_kwargs(
        truncation_method="latent_radius",
        truncation_kwargs={"latent_radius": {"constant_volume_mode": True}},
    )
    assert kwargs == {"latent_radius": {"constant_volume_mode": True}}


def test_normalise_truncation_kwargs_keeps_flat_kwargs_with_methods_list():
    kwargs = normalise_truncation_kwargs(
        truncation_methods=["latent_radius"],
        truncation_kwargs={"constant_volume_mode": True},
    )
    assert kwargs == {"constant_volume_mode": True}


def test_base_truncation_rule_reset_and_getstate():
    rule = DummyTruncationRule()
    rule._value = 2.0
    rule.reset()
    assert np.isnan(rule._value)
    rule._value = 3.0
    state = rule.__getstate__()
    assert np.isnan(state["_value"])


def test_base_truncation_rule_defaults_passthrough():
    rule = BaseTruncationRule()
    x = np.array([1.0])
    log_q = np.array([2.0])
    z = np.array([[3.0]])
    assert rule.configure(None) is None
    assert rule.prepare(None, None) is None
    np.testing.assert_array_equal(rule.apply_latent(None, z), z)
    out = rule.apply_after_backward(None, x, log_q, z)
    assert out == (x, log_q, z)
    out = rule.apply_after_likelihood(None, x, log_q, z)
    assert out == (x, log_q, z)


def test_latent_radius_invalid_radius_mode():
    with pytest.raises(ValueError, match="Unknown radius mode"):
        LatentRadiusTruncation(radius_mode="bad")


def test_latent_radius_fixed_mode_requires_fixed_radius():
    with pytest.raises(ValueError, match="Fixed radius mode requires"):
        LatentRadiusTruncation(radius_mode="fixed")


def test_latent_radius_invalid_fixed_radius_type():
    with pytest.raises(
        RuntimeError, match="fixed_radius must be an int or float"
    ):
        LatentRadiusTruncation(fixed_radius="bad")


def test_latent_radius_configure_constant_volume(proposal):
    proposal.prime_dims = 5
    rule = LatentRadiusTruncation(
        radius_mode="constant_volume",
        max_radius=3.0,
        min_radius=5.0,
        volume_fraction=0.95,
        fuzz=1.5,
    )
    with patch(
        "nessai.proposal.flowproposal.truncation.compute_radius",
        return_value=4.0,
    ) as mock:
        rule.configure(proposal)
    mock.assert_called_once_with(5, 0.95)
    assert rule.fixed_radius == 4.0
    assert rule.min_radius is False
    assert rule.max_radius is False
    assert rule.fuzz == 1.0


def test_latent_radius_configure_expansion_fraction_updates_fuzz(proposal):
    proposal.prime_dims = 4
    rule = LatentRadiusTruncation(expansion_fraction=15.0, fuzz=1.0)
    rule.configure(proposal)
    assert rule.fuzz == pytest.approx(2.0)


def test_latent_radius_compute_radius_with_adaptive_bounds(
    proposal, point, samples
):
    proposal.training_data = samples([(0.0, 0.0), (1.0, 1.0)])
    proposal.forward_pass = lambda *args, **kwargs: (
        np.array([[3.0, 4.0], [6.0, 8.0]]),
        np.array([0.0, 0.0]),
    )
    rule = LatentRadiusTruncation(min_radius=6.0, max_radius=7.0)
    assert rule._compute_radius(proposal, point(0.0, 0.0)) == 7.0


def test_latent_radius_compute_radius_with_all_requires_training_data(
    proposal, point
):
    proposal.training_data = None
    rule = LatentRadiusTruncation(compute_radius_with_all=True)
    with pytest.raises(
        RuntimeError, match="compute_radius_with_all requires training_data"
    ):
        rule._compute_radius(proposal, point(0.0, 0.0))


def test_latent_radius_prepare_and_apply_latent(proposal):
    rule = LatentRadiusTruncation(
        fixed_radius=1.0, radius_mode="fixed", fuzz=2.0
    )
    rule.prepare(proposal, None)
    z = np.array([[0.0, 0.0], [1.5, 0.0], [2.5, 0.0]])
    out = rule.apply_latent(proposal, z)
    assert rule.radius == 1.0
    assert rule.threshold == 2.0
    np.testing.assert_array_equal(out, np.array([[0.0, 0.0], [1.5, 0.0]]))


def test_min_log_q_prepare(proposal, samples):
    proposal.training_data = samples([(0.0, 0.0), (1.0, 1.0)])
    proposal.forward_pass = lambda *args, **kwargs: (
        np.zeros((2, 2)),
        np.array([0.5, -1.5]),
    )
    rule = MinLogQTruncation()
    rule.prepare(proposal, None)
    assert rule.min_log_q == -1.5


def test_min_log_q_prepare_requires_training_data(proposal):
    proposal.training_data = None
    rule = MinLogQTruncation()
    with pytest.raises(
        RuntimeError, match="min_log_q truncation requires training_data"
    ):
        rule.prepare(proposal, None)


def test_min_log_q_filters_after_backward(proposal, samples):
    x = samples([(0.0, 0.0), (1.0, 1.0)])
    log_q = np.array([0.5, -1.5])
    z = np.array([[0.0, 0.0], [1.0, 1.0]])
    rule = MinLogQTruncation()
    rule._min_log_q = -1.0
    out_x, out_log_q, out_z = rule.apply_after_backward(proposal, x, log_q, z)
    assert out_x.size == 1
    assert rule.min_log_q == -1.0
    np.testing.assert_array_equal(out_log_q, np.array([0.5]))
    np.testing.assert_array_equal(out_z, np.array([[0.0, 0.0]]))


def test_likelihood_threshold_properties_and_prepare(point):
    rule = LikelihoodThresholdTruncation()
    assert rule.requires_log_likelihood is True
    rule.prepare(None, point(0.0, 0.0, logl=0.5))
    assert rule.threshold == 0.5


def test_likelihood_threshold_prepare_with_nan_disables_threshold(
    point, caplog
):
    rule = LikelihoodThresholdTruncation()
    with caplog.at_level(logging.DEBUG):
        rule.prepare(None, point(0.0, 0.0, logl=np.nan))
    assert rule.threshold == -np.inf
    assert "Disabling likelihood-threshold truncation" in caplog.text


def test_likelihood_threshold_filters_after_likelihood(proposal, samples):
    x = samples([(0.0, 0.0), (1.0, 1.0)])
    x["logL"] = [0.0, 1.0]
    log_q = np.array([0.5, -1.5])
    z = np.array([[0.0, 0.0], [1.0, 1.0]])
    rule = LikelihoodThresholdTruncation()
    rule._threshold = 0.5
    out_x, out_log_q, out_z = rule.apply_after_likelihood(
        proposal, x, log_q, z
    )
    assert out_x.size == 1
    assert rule.threshold == 0.5
    np.testing.assert_array_equal(out_log_q, np.array([-1.5]))
    np.testing.assert_array_equal(out_z, np.array([[1.0, 1.0]]))


def test_get_truncation_rule_class():
    assert get_truncation_rule_class("latent_radius") is LatentRadiusTruncation


def test_get_truncation_rule_class_unknown():
    with pytest.raises(ValueError, match="Unknown truncation method"):
        get_truncation_rule_class("unknown")


def test_truncation_scheme_duplicate_rule_raises():
    scheme = TruncationScheme([DummyTruncationRule()])
    with pytest.raises(ValueError, match="Duplicate truncation rule"):
        scheme.add_rule(DummyTruncationRule())


def test_truncation_scheme_rule_names_and_has_rule():
    scheme = TruncationScheme(
        [DummyTruncationRule(), LikelihoodThresholdTruncation()]
    )
    assert scheme.rule_names == ["dummy", "likelihood_threshold"]
    assert scheme.has_rule("dummy") is True
    assert scheme.has_rule("missing") is False
    assert scheme.requires_log_likelihood is True
    assert scheme.get_rule("missing") is None


def test_truncation_scheme_add_rule_with_index():
    scheme = TruncationScheme([DummyTruncationRule()])
    scheme.add_rule(LikelihoodThresholdTruncation(), index=0)
    assert scheme.rule_names == ["likelihood_threshold", "dummy"]


def test_truncation_scheme_configure_and_apply_stages(proposal, samples):
    class AddOneRule(BaseTruncationRule):
        name = "add_one"

        def configure(self, proposal) -> None:
            proposal.configured = True

        def apply_latent(self, proposal, z):
            return z + 1.0

    class DropLastRule(BaseTruncationRule):
        name = "drop_last"

        def apply_after_backward(self, proposal, x, log_q, z):
            return x[:-1], log_q[:-1], z[:-1]

    scheme = TruncationScheme([AddOneRule(), DropLastRule()])
    proposal.configured = False
    scheme.configure(proposal)
    assert proposal.configured is True
    np.testing.assert_array_equal(
        scheme.apply_latent(proposal, np.array([[0.0, 0.0]])),
        np.array([[1.0, 1.0]]),
    )
    x = samples([(0.0, 0.0), (1.0, 1.0)])
    log_q = np.array([0.0, 1.0])
    z = np.array([[0.0, 0.0], [1.0, 1.0]])
    out_x, out_log_q, out_z = scheme.apply_after_backward(
        proposal, x, log_q, z
    )
    assert out_x.size == 1
    np.testing.assert_array_equal(out_log_q, np.array([0.0]))
    np.testing.assert_array_equal(out_z, np.array([[0.0, 0.0]]))
    out_x, out_log_q, out_z = scheme.apply_after_likelihood(
        proposal, x, log_q, z
    )
    assert out_x is x
    assert out_log_q is log_q
    assert out_z is z


def test_truncation_scheme_prepare_and_reset(proposal):
    scheme = TruncationScheme(
        [
            LatentRadiusTruncation(
                fixed_radius=2.0,
                radius_mode="fixed",
                volume_fraction=0.9,
                fuzz=1.5,
            ),
            LikelihoodThresholdTruncation(),
        ]
    )
    worst_point = np.zeros(1, dtype=[("logL", "f8")])[0]
    scheme.prepare(proposal, worst_point)
    radius_rule = scheme.get_rule("latent_radius")
    likelihood_rule = scheme.get_rule("likelihood_threshold")
    assert radius_rule.radius == 2.0
    assert radius_rule.threshold == 3.0
    assert radius_rule.volume_fraction == 0.9
    assert likelihood_rule.threshold == 0.0
    scheme.reset()
    assert np.isnan(radius_rule.radius)
    assert np.isnan(radius_rule.threshold)
    assert np.isnan(likelihood_rule.threshold)
