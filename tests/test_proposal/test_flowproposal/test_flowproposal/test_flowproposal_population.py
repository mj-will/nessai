# -*- coding: utf-8 -*-
"""Tests for staged truncation during FlowProposal population."""

import datetime
from unittest.mock import MagicMock

import numpy as np

from nessai.livepoint import get_dtype
from nessai.proposal import FlowProposal
from nessai.proposal.base import PopulationResult
from nessai.proposal.flowproposal.truncation import (
    LatentRadiusTruncation,
    LikelihoodThresholdTruncation,
    MinLogQTruncation,
    TruncationScheme,
)


def get_population_samples(values):
    out = np.zeros(len(values), dtype=get_dtype(["x", "y"]))
    if values:
        out["x"] = [v[0] for v in values]
        out["y"] = [v[1] for v in values]
    return out


def configure_population_test_proposal(proposal, samples):
    proposal.population_time = datetime.timedelta()
    proposal.initialised = True
    proposal.accumulate_weights = True
    proposal.indices = []
    proposal.acceptance = []
    proposal.keep_samples = False
    proposal.check_acceptance = False
    proposal._plot_pool = False
    proposal.populated_count = 0
    proposal.map_to_unit_hypercube = False
    proposal.clip_population_weights = False
    proposal.population_dtype = get_dtype(["x", "y"])
    proposal.convert_to_samples = MagicMock(
        side_effect=lambda x, plot: x.copy()
    )
    proposal.compute_weights = MagicMock(
        side_effect=lambda x, log_q: np.zeros(x.size)
    )
    proposal.rng = MagicMock()
    proposal.rng.random = MagicMock(side_effect=lambda n: np.full(n, 0.5))
    proposal.rng.permutation = MagicMock(side_effect=lambda n: np.arange(n))
    proposal.model = MagicMock()
    proposal.training_data = get_population_samples([(0.0, 0.0), (1.0, 1.0)])
    proposal._truncation_scheme = TruncationScheme()
    proposal.drawsize = 3
    proposal.flow = MagicMock()
    proposal.sample_latent_distribution = MagicMock(
        side_effect=lambda n: proposal.flow.sample_latent_distribution(n)
    )
    proposal._get_population_log_weights = (
        FlowProposal._get_population_log_weights.__get__(
            proposal, FlowProposal
        )
    )
    proposal.record_population_result = MagicMock()
    proposal._pending_model_reset = False
    proposal.last_population_result = None


def test_populate_applies_truncation_in_stages(proposal, point, samples):
    configure_population_test_proposal(proposal, samples)
    proposal._truncation_scheme = TruncationScheme(
        [
            LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed"),
            MinLogQTruncation(),
            LikelihoodThresholdTruncation(),
        ]
    )
    proposal.flow.sample_latent_distribution.return_value = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.5, 0.0]]
    )
    proposal.forward_pass = MagicMock(
        return_value=(np.zeros((2, 2)), np.array([0.0, -1.0]))
    )
    proposal.backward_pass = MagicMock(
        return_value=(
            get_population_samples([(1.0, 1.0), (2.0, 2.0)]),
            np.array([0.5, -2.0]),
            np.array([[0.0, 0.0], [0.5, 0.0]]),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])
    expected_result = PopulationResult(
        completed=True,
        n_requested=1,
        n_proposed=3,
        n_accepted=1,
        population_acceptance=1 / 3,
    )
    proposal.record_population_result.return_value = expected_result

    result = FlowProposal.populate(
        proposal, point(0.0, 0.0, logl=0.5), n_samples=1, plot=False
    )

    proposal.sample_latent_distribution.assert_called_once_with(3)
    proposal.backward_pass.assert_called_once()
    proposal.model.batch_evaluate_log_likelihood.assert_called_once()
    proposal.compute_weights.assert_called_once()
    assert proposal.x.size == 1
    assert proposal.samples.size == 1
    assert proposal._truncation_scheme.get_rule("latent_radius").radius == 1.0
    assert proposal.populated is True
    assert result is expected_result
    proposal.record_population_result.assert_called_once_with(
        completed=True,
        n_requested=1,
        n_proposed=3,
        n_accepted=1,
        hit_max_samples=False,
        request_reset=False,
    )


def test_populate_continues_after_empty_latent_batch(proposal, point, samples):
    configure_population_test_proposal(proposal, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed")]
    )
    proposal.flow.sample_latent_distribution.side_effect = [
        np.array([[2.0, 0.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [2.0, 0.0]]),
    ]
    proposal.backward_pass = MagicMock(
        return_value=(
            get_population_samples([(1.0, 1.0)]),
            np.array([0.5]),
            np.array([[0.0, 0.0]]),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])
    expected_result = PopulationResult(
        completed=True,
        n_requested=1,
        n_proposed=4,
        n_accepted=1,
        population_acceptance=0.25,
    )
    proposal.record_population_result.return_value = expected_result

    result = FlowProposal.populate(
        proposal, point(0.0, 0.0, logl=0.5), n_samples=1, plot=False
    )

    assert proposal.sample_latent_distribution.call_count == 2
    proposal.backward_pass.assert_called_once()
    proposal.model.batch_evaluate_log_likelihood.assert_called_once()
    assert proposal.x.size == 1
    assert result is expected_result
    proposal.record_population_result.assert_called_once_with(
        completed=True,
        n_requested=1,
        n_proposed=4,
        n_accepted=1,
        hit_max_samples=False,
        request_reset=False,
    )


def test_populate_keeps_partial_pool_if_max_samples_hit(
    proposal, point, samples
):
    configure_population_test_proposal(proposal, samples)
    proposal.flow.sample_latent_distribution.return_value = np.zeros((4, 2))
    proposal.backward_pass.return_value = (
        get_population_samples(
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
        ),
        np.zeros(4),
        np.zeros((4, 2)),
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )
    expected_result = PopulationResult(
        completed=False,
        n_requested=5,
        n_proposed=4,
        n_accepted=4,
        population_acceptance=1.0,
        hit_max_samples=True,
    )
    proposal.record_population_result.return_value = expected_result

    result = FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=5,
        plot=False,
        max_samples=3,
    )

    assert proposal.populated is True
    assert proposal.samples.size == 4
    assert proposal.indices == [0, 1, 2, 3]
    assert result is expected_result
    proposal.record_population_result.assert_called_once_with(
        completed=False,
        n_requested=5,
        n_proposed=4,
        n_accepted=4,
        hit_max_samples=True,
        request_reset=True,
    )


def test_populate_returns_empty_result_without_raising(
    proposal, point, samples
):
    configure_population_test_proposal(proposal, samples)
    proposal.flow.sample_latent_distribution.return_value = np.zeros((2, 2))
    proposal.rng.random = MagicMock(side_effect=lambda n: np.ones(n))
    proposal.backward_pass.return_value = (
        get_population_samples([(1.0, 1.0)]),
        np.zeros(1),
        np.zeros((1, 2)),
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])
    expected_result = PopulationResult(
        completed=False,
        n_requested=1,
        n_proposed=2,
        n_accepted=0,
        population_acceptance=0.0,
        hit_max_samples=True,
    )
    proposal.record_population_result.return_value = expected_result

    result = FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=1,
        plot=False,
        max_samples=1,
    )

    assert proposal.populated is False
    assert proposal.samples is None
    assert result is expected_result
    proposal.record_population_result.assert_called_once_with(
        completed=False,
        n_requested=1,
        n_proposed=2,
        n_accepted=0,
        hit_max_samples=True,
        request_reset=True,
    )
