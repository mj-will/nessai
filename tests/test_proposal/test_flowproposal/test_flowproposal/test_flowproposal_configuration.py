# -*- coding: utf-8 -*-
"""Tests for FlowProposal truncation configuration."""

from unittest.mock import MagicMock

import pytest

from nessai.proposal import FlowProposal
from nessai.proposal.flowproposal.truncation import (
    LatentRadiusTruncation,
    LikelihoodThresholdTruncation,
    MinLogQTruncation,
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


def test_configure_truncation_rejects_invalid_fixed_radius(proposal):
    configure_truncation_mocks(proposal)
    with pytest.raises(
        RuntimeError, match="fixed_radius must be an int or float"
    ):
        FlowProposal.configure_truncation(
            proposal,
            latent_radius_kwargs=dict(fixed_radius="bad"),
        )
