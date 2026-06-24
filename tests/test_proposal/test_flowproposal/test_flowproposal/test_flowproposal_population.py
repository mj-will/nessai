# -*- coding: utf-8 -*-
"""Tests for staged truncation during FlowProposal population."""

import datetime
from unittest.mock import MagicMock

import numpy as np

from nessai.livepoint import empty_structured_array
from nessai.proposal import FlowProposal
from nessai.proposal.flowproposal.truncation import (
    LatentRadiusTruncation,
    LikelihoodThresholdTruncation,
    MinLogQTruncation,
    TruncationScheme,
)


def configure_population_test_proposal(proposal, rng, samples):
    proposal.population_time = datetime.timedelta()
    proposal.initialised = True
    proposal.indices = []
    proposal.acceptance = []
    proposal.keep_samples = False
    proposal.check_acceptance = False
    proposal._plot_pool = False
    proposal.populated_count = 0
    proposal.map_to_unit_hypercube = False
    proposal.population_dtype = empty_structured_array(
        0, names=["x", "y"]
    ).dtype
    proposal.convert_to_samples = MagicMock(
        side_effect=lambda x, plot: x.copy()
    )
    proposal.compute_weights = MagicMock(
        side_effect=lambda x, log_q: np.zeros(x.size)
    )
    proposal.rng = MagicMock(wraps=rng)
    proposal.rng.random = MagicMock(side_effect=lambda n: np.full(n, 0.5))
    proposal.rng.permutation = MagicMock(side_effect=lambda n: np.arange(n))
    proposal.model = MagicMock()
    proposal.training_data = samples([(0.0, 0.0), (1.0, 1.0)])
    proposal._truncation_scheme = TruncationScheme()
    proposal.drawsize = 3
    proposal.flow = MagicMock()
    proposal.sample_latent_distribution = MagicMock(
        side_effect=proposal.flow.sample_latent_distribution
    )


def test_populate_applies_truncation_in_stages(proposal, rng, point, samples):
    configure_population_test_proposal(proposal, rng, samples)
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
            samples([(1.0, 1.0), (2.0, 2.0)]),
            np.array([0.5, -2.0]),
            np.array([[0.0, 0.0], [0.5, 0.0]]),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])

    FlowProposal.populate(
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


def test_populate_continues_after_empty_latent_batch(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LatentRadiusTruncation(fixed_radius=1.0, radius_mode="fixed")]
    )
    proposal.flow.sample_latent_distribution.side_effect = [
        np.array([[2.0, 0.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [2.0, 0.0]]),
    ]
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0)]),
            np.array([0.5]),
            np.array([[0.0, 0.0]]),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([1.0])

    FlowProposal.populate(
        proposal, point(0.0, 0.0, logl=0.5), n_samples=1, plot=False
    )

    assert proposal.sample_latent_distribution.call_count == 2
    proposal.backward_pass.assert_called_once()
    proposal.model.batch_evaluate_log_likelihood.assert_called_once()
    assert proposal.x.size == 1


def test_populate_accumulate_weights_recomputes_accept_on_max_samples(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal.accumulate_weights = True
    proposal.flow.sample_latent_distribution.return_value = np.zeros((3, 2))
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            np.zeros(3),
            np.zeros((3, 2)),
        )
    )
    proposal.compute_weights.return_value = np.zeros(3)
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array(
        [1.0, 2.0]
    )

    FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=2,
        plot=False,
        max_samples=2,
    )

    assert proposal.x.size == 2
    assert proposal.samples.size == 2
    assert proposal.sample_latent_distribution.call_count == 1
    assert proposal.rng.random.call_count == 1


def test_populate_stops_at_max_samples_after_empty_latent_batches(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LatentRadiusTruncation(fixed_radius=0.1, radius_mode="fixed")]
    )
    proposal.flow.sample_latent_distribution.return_value = np.array(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.array([])

    FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=1,
        plot=False,
        max_samples=2,
    )

    proposal.sample_latent_distribution.assert_called_once_with(3)
    assert proposal.backward_pass.call_count == 0
    assert proposal.x.size == 0
    assert proposal.population_acceptance == 0.0


def test_populate_stops_at_max_samples_after_all_likelihood_rejected(
    proposal, rng, point, samples
):
    configure_population_test_proposal(proposal, rng, samples)
    proposal._truncation_scheme = TruncationScheme(
        [LikelihoodThresholdTruncation()]
    )
    proposal.flow.sample_latent_distribution.return_value = np.zeros((3, 2))
    proposal.backward_pass = MagicMock(
        return_value=(
            samples([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            np.zeros(3),
            np.zeros((3, 2)),
        )
    )
    proposal.model.batch_evaluate_log_likelihood.return_value = np.zeros(3)

    FlowProposal.populate(
        proposal,
        point(0.0, 0.0, logl=0.5),
        n_samples=1,
        plot=False,
        max_samples=2,
    )

    proposal.sample_latent_distribution.assert_called_once_with(3)
    proposal.backward_pass.assert_called_once()
    proposal.compute_weights.assert_not_called()
    assert proposal.x.size == 0
    assert proposal.population_acceptance == 0.0
