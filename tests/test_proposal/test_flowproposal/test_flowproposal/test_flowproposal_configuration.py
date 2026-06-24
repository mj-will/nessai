# -*- coding: utf-8 -*-
"""Tests for FlowProposal truncation configuration."""

import logging
import warnings
from unittest.mock import MagicMock

import pytest

from nessai.proposal import FlowProposal
from nessai.proposal.flowproposal.truncation import (
    BaseTruncationRule,
    LatentRadiusTruncation,
    LikelihoodThresholdTruncation,
    MinLogQTruncation,
    TruncationScheme,
)


def configure_truncation_mocks(proposal, latent_radius_rule=None):
    proposal._get_latent_radius_rule = MagicMock(
        return_value=latent_radius_rule
    )
    proposal._sync_truncation_state = MagicMock(return_value=None)


def test_configure_population_sets_defaults(proposal):
    proposal.poolsize = 2000
    FlowProposal.configure_population(proposal, None)
    assert proposal.drawsize == 2000
    assert proposal.latent_prior == "flow"
    assert proposal.latent_temperature is None


def test_configure_population_rejects_non_flow_prior(proposal):
    with pytest.raises(ValueError, match="Only the flow latent prior"):
        FlowProposal.configure_population(
            proposal, 1000, latent_prior="gaussian"
        )


def test_configure_population_sets_latent_temperature(proposal):
    FlowProposal.configure_population(proposal, 1000, latent_temperature=0.9)
    assert proposal.latent_temperature == 0.9


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_configure_population_rejects_non_positive_latent_temperature(
    proposal, temperature
):
    with pytest.raises(
        ValueError, match="latent_temperature must be positive"
    ):
        FlowProposal.configure_population(
            proposal, 1000, latent_temperature=temperature
        )


def test_configure_population_rejects_invalid_latent_temperature_type(
    proposal,
):
    with pytest.raises(TypeError, match="latent_temperature must be a float"):
        FlowProposal.configure_population(
            proposal, 1000, latent_temperature="bad"
        )


def test_truncation_property_and_rule_helpers(proposal):
    proposal._truncation_scheme = TruncationScheme(
        [LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed")]
    )
    proposal.truncation = proposal._truncation_scheme
    assert (
        FlowProposal.truncation.fget(proposal) is proposal._truncation_scheme
    )
    assert isinstance(
        FlowProposal.get_truncation_rule(proposal, "latent_radius"),
        LatentRadiusTruncation,
    )
    assert isinstance(
        FlowProposal._get_latent_radius_rule(proposal),
        LatentRadiusTruncation,
    )


def test_sync_truncation_state_updates_legacy_flags(proposal):
    proposal._truncation_scheme = TruncationScheme(
        [MinLogQTruncation(), LikelihoodThresholdTruncation()]
    )
    FlowProposal._sync_truncation_state(proposal)
    assert proposal.truncation_methods == [
        "min_log_q",
        "likelihood_threshold",
    ]
    assert proposal.truncate_log_q is True
    assert proposal.enforce_likelihood_threshold is True


def test_warning_helpers_do_nothing_when_disabled():
    with warnings.catch_warnings(record=True) as record:
        FlowProposal._warn_for_deprecated_latent_prior(None)
        FlowProposal._warn_for_deprecated_likelihood_threshold(False)
        FlowProposal._warn_for_deprecated_truncation_arguments(
            MagicMock(),
            fixed_radius=None,
            radius_mode=None,
            min_radius=None,
            max_radius=None,
            compute_radius_with_all=None,
            constant_volume_mode=None,
            volume_fraction=None,
            fuzz=None,
            expansion_fraction=None,
        )
    assert not record


def test_log_configuration_logs_rules(proposal, caplog):
    class DummyRule(BaseTruncationRule):
        name = "dummy"

    proposal.drawsize = 10
    proposal.latent_prior = "flow"
    proposal.latent_temperature = None
    proposal.truncation_methods = ["latent_radius", "dummy"]
    proposal._truncation_scheme = TruncationScheme(
        [
            LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed"),
            DummyRule(),
        ]
    )
    with caplog.at_level(logging.INFO):
        FlowProposal._log_configuration(proposal)
    assert "FlowProposal configuration" in caplog.text
    assert "FlowProposal truncation rule latent_radius" in caplog.text
    assert "FlowProposal truncation rule enabled: dummy" in caplog.text


def test_configure_truncation_normalises_string(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal, truncation_method="likelihood_threshold"
    )
    assert proposal._truncation_scheme.rule_names == ["likelihood_threshold"]
    proposal._sync_truncation_state.assert_called_once_with()
    assert isinstance(
        proposal._truncation_scheme.get_rule("likelihood_threshold"),
        LikelihoodThresholdTruncation,
    )


def test_configure_truncation_deduplicates_methods(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        truncation_methods=["min_log_q", "min_log_q", "likelihood_threshold"],
    )
    assert proposal._truncation_scheme.rule_names == [
        "min_log_q",
        "likelihood_threshold",
    ]
    proposal._sync_truncation_state.assert_called_once_with()


def test_configure_truncation_unknown_method(proposal):
    configure_truncation_mocks(proposal)
    with pytest.raises(ValueError, match="Unknown truncation method"):
        FlowProposal.configure_truncation(
            proposal, truncation_methods=["unknown"]
        )


def test_configure_truncation_rejects_method_and_methods(proposal):
    configure_truncation_mocks(proposal)
    with pytest.raises(
        ValueError, match="Specify only one of truncation_method"
    ):
        FlowProposal.configure_truncation(
            proposal,
            truncation_method="min_log_q",
            truncation_methods=["likelihood_threshold"],
        )


def test_configure_truncation_enables_legacy_flags(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        truncate_log_q=True,
        enforce_likelihood_threshold=True,
    )
    assert proposal._truncation_scheme.rule_names == [
        "min_log_q",
        "likelihood_threshold",
    ]
    proposal._sync_truncation_state.assert_called_once_with()
    assert isinstance(
        proposal._truncation_scheme.get_rule("min_log_q"), MinLogQTruncation
    )


def test_configure_truncation_enables_latent_radius_from_kwargs(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        latent_radius_kwargs=dict(fixed_radius=4.0, radius_mode="fixed"),
    )
    assert proposal._truncation_scheme.rule_names == ["latent_radius"]
    proposal._sync_truncation_state.assert_called_once_with()
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert isinstance(rule, LatentRadiusTruncation)
    assert rule.fixed_radius == 4.0
    assert rule.radius_mode == "fixed"


def test_configure_truncation_creates_radius_rule(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        latent_radius_kwargs=dict(fixed_radius=5.0, radius_mode="fixed"),
    )
    assert proposal._truncation_scheme.rule_names == ["latent_radius"]
    proposal._sync_truncation_state.assert_called_once_with()
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.fixed_radius == 5.0
    assert rule.radius_mode == "fixed"


def test_configure_truncation_applies_default_radius_kwargs(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        default_latent_radius=True,
    )
    assert proposal._truncation_scheme.rule_names == ["latent_radius"]
    proposal._sync_truncation_state.assert_called_once_with()
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.radius_mode == "constant_volume"
    assert rule.volume_fraction == 0.95


def test_configure_truncation_does_not_apply_default_radius_to_explicit_methods(
    proposal,
):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        truncation_methods=["min_log_q"],
        default_latent_radius=True,
    )
    assert proposal._truncation_scheme.rule_names == ["min_log_q"]
    proposal._sync_truncation_state.assert_called_once_with()
    assert proposal._truncation_scheme.get_rule("latent_radius") is None


def test_configure_truncation_explicit_empty_list_disables_default_radius(
    proposal,
):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        truncation_methods=[],
        default_latent_radius=True,
    )
    assert proposal._truncation_scheme.rule_names == []
    proposal._sync_truncation_state.assert_called_once_with()
    assert proposal._truncation_scheme.get_rule("latent_radius") is None


def test_configure_truncation_enables_radius_from_expansion_fraction(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        latent_radius_kwargs=dict(expansion_fraction=None),
    )
    assert proposal._truncation_scheme.rule_names == ["latent_radius"]
    proposal._sync_truncation_state.assert_called_once_with()
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.expansion_fraction is None


def test_configure_truncation_constant_volume_updates_rule(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        latent_radius_kwargs=dict(fixed_radius=5.0, constant_volume_mode=True),
    )
    assert proposal._truncation_scheme.rule_names == ["latent_radius"]
    proposal._sync_truncation_state.assert_called_once_with()
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.constant_volume_mode is True
    assert rule.radius_mode == "constant_volume"


def test_configure_truncation_supports_flat_kwargs_for_single_method(proposal):
    configure_truncation_mocks(proposal)
    FlowProposal.configure_truncation(
        proposal,
        truncation_method="latent_radius",
        truncation_kwargs=dict(constant_volume_mode=True),
    )
    assert proposal._truncation_scheme.rule_names == ["latent_radius"]
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.constant_volume_mode is True
    assert rule.radius_mode == "constant_volume"


def test_configure_truncation_reuses_existing_latent_radius_rule(proposal):
    existing_rule = LatentRadiusTruncation(
        fixed_radius=3.0,
        radius_mode="fixed",
        fuzz=1.2,
    )
    configure_truncation_mocks(proposal, latent_radius_rule=existing_rule)
    FlowProposal.configure_truncation(proposal)
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.fixed_radius == 3.0
    assert rule.radius_mode == "fixed"
    assert rule.fuzz == 1.2


def test_configure_truncation_rejects_non_dict_truncation_kwargs(proposal):
    configure_truncation_mocks(proposal)
    with pytest.raises(
        TypeError,
        match="Truncation kwargs for latent_radius must be a dictionary",
    ):
        FlowProposal.configure_truncation(
            proposal,
            truncation_method="latent_radius",
            truncation_kwargs={"latent_radius": 1.0},
        )


def test_configure_truncation_rejects_invalid_fixed_radius(proposal):
    configure_truncation_mocks(proposal)
    with pytest.raises(
        RuntimeError, match="fixed_radius must be an int or float"
    ):
        FlowProposal.configure_truncation(
            proposal,
            latent_radius_kwargs=dict(fixed_radius="bad"),
        )
