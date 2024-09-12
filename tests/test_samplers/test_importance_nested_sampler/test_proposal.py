"""Tests for the proposals and meta-proposal"""

import datetime
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai.proposal.importance import ImportanceFlowProposal
from nessai.samplers.importancesampler import (
    ImportanceNestedSampler as INS,
)
from nessai.samplers.importancesampler import (
    OrderedSamples,
)
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture()
def ins(ins, proposal):
    ins.proposal = proposal
    return ins


def test_get_proposal(ins, model, tmp_path):
    output = str(tmp_path)
    subdir = "proposal"
    ins.model = model
    ins.output = output
    instance = object()
    with patch(
        "nessai.samplers.importancesampler.ImportanceFlowProposal",
        return_value=instance,
    ) as mock_class:
        out = INS.get_proposal(ins, subdir, test=1)
    assert out is instance
    mock_class.assert_called_once_with(
        model,
        os.path.join(output, subdir, ""),
        test=1,
    )


@pytest.mark.parametrize("weighted_kl", [False, True])
def test_add_new_proposal(ins, samples, log_q, weighted_kl):
    n = int(0.8 * len(samples))

    ins.training_samples = MagicMock(spec=OrderedSamples)
    ins.training_samples.samples = np.sort(samples, order="logL")
    ins.training_samples.log_q = log_q
    ins.log_likelihood_threshold = ins.training_samples.samples[n]["logL"]
    ins.min_samples = 2

    ins.replace_all = False
    ins.weighted_kl = weighted_kl
    ins.plot_training_data = True
    ins.training_time = datetime.timedelta(seconds=10)

    INS.add_new_proposal(ins)

    ins.proposal.train.assert_called_once()
    assert_structured_arrays_equal(
        ins.proposal.train.call_args_list[0][0][0],
        ins.training_samples.samples[n:],
    )


def test_draw_n_samples(ins, samples, log_q, history):
    expected = samples.copy()
    n = len(expected)
    ins.draw_samples_time = datetime.timedelta(seconds=10)
    ins.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=samples["logL"].copy()
    )
    samples["logL"] = np.nan
    ins.proposal.draw = MagicMock(return_value=(samples, log_q))

    out = INS.draw_n_samples(ins, n)

    ins.proposal.draw.assert_called_once_with(n)

    np.testing.assert_array_equal(out[1], log_q)
    assert_structured_arrays_equal(out[0], expected)


def test_update_proposal_weights(ins):
    ins.samples_unit = np.ones(10)
    ins.sample_counts = {-1: 2, 0: 4, 1: 4}
    ins.proposal = MagicMock(spec=ImportanceFlowProposal)
    INS.update_proposal_weights(ins)
    expected_weights = {-1: 0.2, 0: 0.4, 1: 0.4}
    ins.proposal.update_proposal_weights.assert_called_once_with(
        expected_weights
    )


def test_add_new_proposal_weight(ins):
    n = 8
    n_new = 2
    sample_counts = {-1: 2, 0: 3, 1: 3}
    iteration = 2

    ins.samples_unit = np.ones(n)
    ins.sample_counts = sample_counts
    ins.proposal = MagicMock(spec=ImportanceFlowProposal)

    INS.add_new_proposal_weight(ins, iteration, n_new)

    assert ins.sample_counts[2] == 2
    expected_weights = {-1: 0.2, 0: 0.3, 1: 0.3, 2: 0.2}
    ins.proposal.update_proposal_weights.assert_called_once_with(
        expected_weights
    )


def test_add_new_proposal_weight_error(ins):
    n = 8
    n_new = 2
    sample_counts = {-1: 2, 0: 3, 1: 3, 2: 2}
    iteration = 2

    ins.samples_unit = np.ones(n)
    ins.sample_counts = sample_counts

    with pytest.raises(
        RuntimeError, match="Samples already drawn from proposal 2"
    ):
        INS.add_new_proposal_weight(ins, iteration, n_new)
