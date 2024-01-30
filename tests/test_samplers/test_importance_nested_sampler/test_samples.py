"""Tests related to how samples are handled"""
from unittest.mock import MagicMock, create_autospec

from nessai.livepoint import numpy_array_to_live_points
from nessai.samplers.importancesampler import (
    ImportanceNestedSampler as INS,
    OrderedSamples,
)
import numpy as np


def test_populate_live_points_no_iid(ins, model):
    n = 100
    ins.n_initial = n
    ins.model = model
    ins.draw_iid_live = False

    INS.populate_live_points(ins)

    ins.training_samples.add_initial_samples.assert_called_once()
    assert len(ins.training_samples.add_initial_samples.call_args.args[0]) == n
    assert ins.training_samples.add_initial_samples.call_args.args[
        1
    ].shape == (n, 1)


def test_populate_live_points_iid(ins, model):
    n = 100
    ins.n_initial = n
    ins.model = model
    ins.draw_iid_live = True
    ins.iid_samples = create_autospec(OrderedSamples)

    INS.populate_live_points(ins)

    ins.training_samples.add_initial_samples.assert_called_once()
    assert len(ins.training_samples.add_initial_samples.call_args.args[0]) == n
    assert ins.training_samples.add_initial_samples.call_args.args[
        1
    ].shape == (n, 1)
    assert np.isfinite(
        ins.training_samples.add_initial_samples.call_args.args[0]["logL"]
    ).all()
    assert np.isfinite(
        ins.training_samples.add_initial_samples.call_args.args[0]["logP"]
    ).all()

    ins.iid_samples.add_initial_samples.assert_called_once()
    assert len(ins.iid_samples.add_initial_samples.call_args.args[0]) == n
    assert ins.iid_samples.add_initial_samples.call_args.args[1].shape == (
        n,
        1,
    )
    assert np.isfinite(
        ins.iid_samples.add_initial_samples.call_args.args[0]["logL"]
    ).all()
    assert np.isfinite(
        ins.iid_samples.add_initial_samples.call_args.args[0]["logP"]
    ).all()


def test_remove_samples(ins, iid):
    ins.history = {}
    ins.history["n_removed"] = [3]
    ins.training_samples.remove_samples = MagicMock(return_value=5)
    ins.draw_iid_live = iid
    expected = 5
    if iid:
        ins.iid_samples = create_autospec(OrderedSamples)
        ins.iid_samples.remove_samples = MagicMock(return_value=4)
        expected = 4

    out = INS.remove_samples(ins)

    ins.training_samples.remove_samples.assert_called_once()
    assert out == expected
    assert ins.history["n_removed"] == [3, expected]


def test_adjust_final_samples(ins, proposal, model, samples, log_q):
    def draw(n, flow_number=None, update_counts=False):
        assert update_counts is False
        x = numpy_array_to_live_points(
            np.random.randn(n, model.dims),
            names=model.names,
        )
        lq = np.random.rand(n, log_q.shape[1])
        return x, lq

    def draw_from_prior(n):
        x = model.new_point(n)
        lq = np.random.rand(n, log_q.shape[1])
        return x, lq

    proposal.draw = MagicMock(side_effect=draw)
    proposal.draw_from_prior = MagicMock(side_effect=draw_from_prior)
    proposal.n_requested = {"-1": 10, "1": 10}

    ins.samples_unit = samples
    ins.log_q = log_q
    ins.proposal = proposal
    ins.model = model

    INS.adjust_final_samples(ins)
