# -*- coding: utf-8 -*-
"""Tests for initialising, pickling, and resetting FlowProposal."""

from unittest.mock import patch

import numpy as np
import pytest

from nessai.proposal import FlowProposal


def test_init(model):
    with pytest.warns(None) as record:
        fp = FlowProposal(model, poolsize=1000)
    assert not record
    assert fp.model == model
    assert fp.poolsize == 1000
    assert fp.latent_prior == "flow"
    assert fp.truncation_methods == ["latent_radius"]
    rule = fp.get_truncation_rule("latent_radius")
    assert rule.constant_volume_mode is True
    assert rule.volume_fraction == 0.95


def test_init_with_explicit_flow_latent_prior_warns(model):
    with pytest.warns(FutureWarning, match="latent_prior"):
        fp = FlowProposal(model, poolsize=1000, latent_prior="flow")
    assert fp.latent_prior == "flow"


def test_init_with_truncation_methods(model):
    fp = FlowProposal(
        model,
        poolsize=1000,
        truncation_methods=["min_log_q", "likelihood_threshold"],
    )
    assert fp.truncation_methods == [
        "min_log_q",
        "likelihood_threshold",
    ]
    assert fp._truncation_scheme.rule_names == [
        "min_log_q",
        "likelihood_threshold",
    ]


def test_init_with_radius_configuration(model):
    with pytest.warns(
        FutureWarning, match="truncation_kwargs\\['latent_radius'\\]"
    ):
        fp = FlowProposal(
            model,
            poolsize=1000,
            fixed_radius=5.0,
            radius_mode="fixed",
            volume_fraction=0.8,
        )
    assert fp.truncation_methods == ["latent_radius"]
    rule = fp.get_truncation_rule("latent_radius")
    assert rule.radius_mode == "fixed"
    assert rule.fixed_radius == 5.0
    assert rule.volume_fraction == 0.8


def test_init_with_truncation_kwargs_does_not_warn(model):
    with pytest.warns(None) as record:
        fp = FlowProposal(
            model,
            poolsize=1000,
            truncation_methods=["latent_radius"],
            truncation_kwargs={
                "latent_radius": {
                    "fixed_radius": 5.0,
                    "radius_mode": "fixed",
                    "volume_fraction": 0.8,
                }
            },
        )
    assert not record
    assert fp.truncation_methods == ["latent_radius"]


def test_deprecated_latent_radius_arguments_warn(model):
    with pytest.warns(FutureWarning) as record:
        FlowProposal(
            model,
            poolsize=1000,
            fixed_radius=5.0,
            radius_mode="fixed",
            min_radius=1.0,
            max_radius=10.0,
            compute_radius_with_all=True,
            constant_volume_mode=True,
            volume_fraction=0.8,
            fuzz=1.2,
            expansion_fraction=1.2,
        )
    message = str(record[0].message)
    for name in (
        "fixed_radius",
        "radius_mode",
        "min_radius",
        "max_radius",
        "compute_radius_with_all",
        "constant_volume_mode",
        "volume_fraction",
        "fuzz",
        "expansion_fraction",
    ):
        assert name in message


def test_deprecated_enforce_likelihood_threshold_warns(model):
    with pytest.warns(FutureWarning, match="enforce_likelihood_threshold"):
        FlowProposal(
            model,
            poolsize=1000,
            enforce_likelihood_threshold=True,
        )


def test_init_rejects_non_flow_latent_prior(model):
    with pytest.raises(ValueError, match="Only the flow latent prior"):
        FlowProposal(model, poolsize=1000, latent_prior="gaussian")


def test_pickle_with_truncation_methods(model, tmpdir):
    import pickle

    proposal = FlowProposal(
        model,
        poolsize=1000,
        plot=False,
        output=tmpdir.mkdir("truncation_pickle"),
        truncation_methods=["min_log_q", "likelihood_threshold"],
    )
    proposal._truncation_scheme.get_rule("min_log_q")._min_log_q = -10.0
    proposal._truncation_scheme.get_rule(
        "likelihood_threshold"
    )._threshold = 1.0
    proposal_re = pickle.loads(pickle.dumps(proposal))
    assert proposal_re.truncation_methods == [
        "min_log_q",
        "likelihood_threshold",
    ]
    assert proposal_re._truncation_scheme.rule_names == [
        "min_log_q",
        "likelihood_threshold",
    ]
    assert np.isnan(
        proposal_re._truncation_scheme.get_rule("min_log_q").min_log_q
    )
    assert np.isnan(
        proposal_re._truncation_scheme.get_rule(
            "likelihood_threshold"
        ).threshold
    )


def test_pickle_with_radius_rule(model, tmpdir):
    import pickle

    with pytest.warns(
        FutureWarning, match="truncation_kwargs\\['latent_radius'\\]"
    ):
        proposal = FlowProposal(
            model,
            poolsize=1000,
            plot=False,
            output=tmpdir.mkdir("radius_pickle"),
            fixed_radius=5.0,
            radius_mode="fixed",
        )
    proposal._truncation_scheme.get_rule("latent_radius")._radius = 3.0
    proposal._truncation_scheme.get_rule("latent_radius")._threshold = 4.0
    proposal_re = pickle.loads(pickle.dumps(proposal))
    assert proposal_re.truncation_methods == ["latent_radius"]
    rule = proposal_re.get_truncation_rule("latent_radius")
    assert rule.fixed_radius == 5.0
    assert rule.radius_mode == "fixed"
    assert np.isnan(rule.radius)
    assert np.isnan(rule.threshold)


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("mask", [None, [1, 0]])
def test_get_state(proposal, populated, mask):
    parent_state = {"a": "val"}
    with patch(
        "nessai.proposal.flowproposal.base.BaseFlowProposal.__getstate__",
        return_value=parent_state,
    ) as mock:
        state = FlowProposal.__getstate__(proposal)

    mock.assert_called_once()
    assert state["a"] == "val"


def test_reset(proposal):
    FlowProposal.configure_truncation(
        proposal,
        latent_radius_kwargs=dict(fixed_radius=2.0, radius_mode="fixed"),
    )
    proposal.get_truncation_rule("latent_radius")._radius = 1.0
    with patch(
        "nessai.proposal.flowproposal.base.BaseFlowProposal.reset"
    ) as mock:
        FlowProposal.reset(proposal)
    mock.assert_called_once()
    assert np.isnan(proposal.get_truncation_rule("latent_radius").radius)
