"""Test configuration of INS"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai import config as nessai_config
from nessai.samplers.importancesampler import ImportanceNestedSampler as INS


@pytest.mark.parametrize("save_log_q", [False, True])
def test_init(ins, model, save_log_q):
    ins.rng = None
    model.rng = None
    ins.add_fields = MagicMock()
    INS.__init__(ins, model, save_log_q=save_log_q, draw_iid_live=True)
    ins.add_fields.assert_called_once()
    assert ins.training_samples.save_log_q is save_log_q
    assert ins.iid_samples.save_log_q is save_log_q


def test_add_fields():
    INS.add_fields()
    assert "logW" in nessai_config.livepoints.non_sampling_parameters
    assert "logQ" in nessai_config.livepoints.non_sampling_parameters
    assert "logU" in nessai_config.livepoints.non_sampling_parameters


@pytest.mark.parametrize("it, expected", [(None, -1), (100, 100)])
def test_configure_min_iterations(ins, it, expected):
    INS.configure_iterations(ins, min_iteration=it)
    assert ins.min_iteration == expected


@pytest.mark.parametrize("it, expected", [(None, np.inf), (100, 100)])
def test_configure_max_iterations(ins, it, expected):
    INS.configure_iterations(ins, max_iteration=it)
    assert ins.max_iteration == expected


def test_initialise(ins):
    ins.initialised = False
    ins.live_points_unit = None
    ins.populate_live_points = MagicMock()
    ins.proposal = MagicMock()

    INS.initialise(ins)

    ins.populate_live_points.assert_called_once()
    ins.proposal.initialise.assert_called_once()
    assert ins.initialised is True


def test_initialise_history(ins):
    ins.history = None
    INS.initialise_history(ins)
    assert ins.history is not None


def test_check_configuration_min_samples(ins):
    ins.min_samples = 100
    ins.nlive = 10
    ins.min_remove = 1
    with pytest.raises(
        ValueError, match="`min_samples` must be less than `nlive`"
    ):
        INS.check_configuration(ins)


def test_check_configuration_min_remove(ins):
    ins.min_samples = 50
    ins.nlive = 100
    ins.min_remove = 200
    with pytest.raises(
        ValueError, match="`min_remove` must be less than `nlive`"
    ):
        INS.check_configuration(ins)


def check_configuration_okay(ins):
    ins.min_samples = 50
    ins.nlive = 100
    ins.min_remove = 1
    assert INS.check_configuration(ins) is True


@pytest.mark.parametrize("has_proposal", [False, True])
def test_update_output(ins, tmp_path, has_proposal):
    output = tmp_path / "new"
    if has_proposal:
        ins.proposal = MagicMock()
        ins.proposal.output = tmp_path / "orig" / "levels"
    else:
        ins.proposal = None
    with patch("nessai.samplers.base.BaseNestedSampler.update_output") as mock:
        INS.update_output(ins, output)

    mock.assert_called_once_with(output)
    if has_proposal:
        ins.proposal.update_output.assert_called_once_with(
            os.path.join(output, "levels", "")
        )
