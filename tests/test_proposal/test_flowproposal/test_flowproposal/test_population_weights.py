"""Tests for clipping and normalising population weights."""

import numpy as np
import pytest

from nessai.proposal import FlowProposal
from nessai.proposal.flowproposal.flowproposal import _clip_weights


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([], []),
        ([4.0], [1.0]),
        ([1.0, 9.0], [1.0, 1.0]),
        ([1.0, 9.0, 3.0, 7.0], [0.2, 1.6, 0.6, 1.6]),
        ([1.0, 12.0, 3.0, 6.0, 8.0], [1 / 6, 13 / 9, 0.5, 13 / 9, 13 / 9]),
        ([2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_clip_weights(weights, expected):
    weights = np.asarray(weights, dtype=float)
    original = weights.copy()

    result = _clip_weights(weights)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(weights, original)
    assert not np.shares_memory(result, weights)


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("offset", [-1000.0, 0.0, 1000.0])
def test_population_log_weights(proposal, clip, offset):
    proposal.clip_population_weights = clip
    log_weights = np.log([1.0, 9.0, 3.0, 7.0]) + offset
    original = log_weights.copy()
    expected = [1 / 8, 1.0, 3 / 8, 1.0] if clip else [1 / 9, 1, 1 / 3, 7 / 9]

    result = FlowProposal._get_population_log_weights(proposal, log_weights)

    np.testing.assert_allclose(result, np.log(expected), atol=1e-12)
    np.testing.assert_array_equal(log_weights, original)
    assert result.max() == 0.0


@pytest.mark.parametrize("clip", [False, True])
def test_population_log_weights_empty(proposal, clip):
    proposal.clip_population_weights = clip

    result = FlowProposal._get_population_log_weights(proposal, [])

    assert result.shape == (0,)
    assert result.dtype == float
