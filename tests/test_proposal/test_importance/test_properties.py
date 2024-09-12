"""Test the properties defined in ImportanceFlowProposal"""

from unittest.mock import patch

import numpy as np
import pytest

from nessai.proposal.importance import ImportanceFlowProposal as IFP


@pytest.fixture
def weights():
    return {"-1": 0.2, "0": 0.3, "1": 0.5}


@pytest.fixture
def ifp(ifp, weights):
    ifp._weights = weights
    return ifp


def test_weights(ifp, weights):
    assert IFP.weights.__get__(ifp) == weights


def test_weights_array(ifp, weights):
    np.testing.assert_array_equal(
        IFP.weights_array.__get__(ifp), np.array(list(weights.values()))
    )


def test_n_proposals(ifp):
    IFP.n_proposals.__get__(ifp) == 3


def test_flow_config(ifp):
    flow_config = {"a": 1}
    ifp._flow_config = flow_config
    IFP.flow_config.__get__(ifp) is flow_config


@pytest.mark.parametrize("config", [None, {}, {"model_config": {}}])
def test_flow_config_setter(ifp, config):
    ifp.model.dims = 2
    out = {"test": True, "model_config": {"n_inputs": 2}}
    expected = (
        config | {"n_inputs": 2} if config is not None else {"n_inputs": 2}
    )
    with patch(
        "nessai.proposal.importance.update_flow_config", return_value=out
    ) as mock:
        IFP.flow_config.__set__(ifp, config)
    mock.assert_called_once_with(expected)
    assert ifp._flow_config is out


@pytest.mark.parametrize(
    "reset_flow, level_count",
    [[1, 3], [4, 8], [True, 5]],
)
def test_reset_flow_true(ifp, reset_flow, level_count):
    ifp.reset_flow = reset_flow
    ifp.level_count = level_count
    assert IFP._reset_flow.__get__(ifp) is True


@pytest.mark.parametrize(
    "reset_flow, level_count",
    [[False, 4], [5, 8]],
)
def test_reset_flow_false(ifp, reset_flow, level_count):
    ifp.reset_flow = reset_flow
    ifp.level_count = level_count
    assert IFP._reset_flow.__get__(ifp) is False
