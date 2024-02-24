"""Test the properties defined in ImportanceFlowProposal"""

from unittest.mock import patch
from nessai.proposal.importance import ImportanceFlowProposal as IFP
import pytest


@pytest.fixture
def n_draws():
    return {"-1": 10, "0": 20, "1": 30}


@pytest.fixture
def ifp(ifp, n_draws):
    ifp.n_draws = n_draws
    return ifp


def test_total_samples_drawn(ifp):
    assert IFP.total_samples_drawn.__get__(ifp) == 60


def test_unnormalised_weights(ifp, n_draws):
    assert IFP.unnormalised_weights.__get__(ifp) == n_draws


def test_poolsize(ifp, n_draws):
    ifp.unnormalised_weights = n_draws
    IFP.poolsize.__get__(ifp) == [10, 20, 30]


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
    with patch(
        "nessai.proposal.importance.update_config", return_value=out
    ) as mock:
        IFP.flow_config.__set__(ifp, config)
    mock.assert_called_once_with({"model_config": {"n_inputs": 2}})
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
