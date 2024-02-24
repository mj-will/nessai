"""Test the general configuration for ImportanceFlowProposal"""

import pytest
from unittest.mock import MagicMock, patch

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


@pytest.mark.usefixtures("ins_parameters")
def test_check_fields_pass():
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


def test_initialise(ifp):
    output = "test_init"
    flow_config = {"a": 1}
    ifp.initialised = False
    ifp.output = output
    ifp.flow_config = flow_config
    flow = MagicMock()
    with patch(
        "nessai.proposal.importance.Proposal.initialise",
    ) as mock_init, patch(
        "nessai.proposal.importance.ImportanceFlowModel", return_value=flow
    ) as mock_flow:
        IFP.initialise(ifp)
    ifp._check_fields.assert_called_once()
    ifp.verify_rescaling.assert_called_once()
    # Make sure flow is initialised
    mock_flow.assert_called_once_with(config=flow_config, output=output)
    flow.initialise.assert_called_once()
    mock_init.assert_called_once()


def test_already_initialised(ifp):
    """Assert functions are not called"""
    ifp.initialised = True

    with patch(
        "nessai.proposal.importance.Proposal.initialise",
    ) as mock_init, patch(
        "nessai.proposal.importance.ImportanceFlowModel",
    ) as mock_flow:
        IFP.initialise(ifp)
    mock_init.assert_not_called()
    mock_flow.assert_not_called()
    ifp._check_fields.assert_called_once()
    ifp.verify_rescaling.assert_not_called()
