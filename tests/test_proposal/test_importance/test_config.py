"""Test the general configuration for ImportanceFlowProposal"""
import pytest

from nessai.livepoint import (
    add_extra_parameters_to_live_points,
    reset_extra_live_points_parameters,
)
from nessai.proposal.importance import ImportanceFlowProposal as IFP


def test_init(ifp, model, tmp_path):
    """Test the init method"""
    output = tmp_path / "test"
    initial_draws = 1000
    IFP.__init__(ifp, model, output, initial_draws)

    assert ifp.n_draws["initial"] == initial_draws
    assert ifp.n_requested["initial"] == initial_draws


def test_check_fields_pass(ins_parameters):
    """Assert the checks pass"""
    IFP._check_fields()


def test_check_fields_logQ():
    """Assert an error is raised if logQ is missing"""
    with pytest.raises(
        RuntimeError, match=r"logQ field missing in non-sampling parameters."
    ):
        IFP._check_fields()


def test_check_fields_logW():
    """Assert an error is raised if logW is missing"""
    add_extra_parameters_to_live_points(["logQ"])
    with pytest.raises(
        RuntimeError, match=r"logW field missing in non-sampling parameters."
    ):
        IFP._check_fields()
    reset_extra_live_points_parameters()
