"""Test the general configuration for ImportanceFlowProposal"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai.livepoint import (
    add_extra_parameters_to_live_points,
    reset_extra_live_points_parameters,
)
from nessai.proposal.importance import ImportanceFlowProposal as IFP


@pytest.mark.parametrize("rng", [None, np.random.RandomState()])
def test_init(ifp, model, tmp_path, rng):
    """Test the init method"""
    output = tmp_path / "test"
    mock_rng = MagicMock
    with patch(
        "numpy.random.default_rng", return_value=mock_rng
    ) as mock_default_rng:
        IFP.__init__(ifp, model=model, output=output, rng=rng)
    assert ifp._weights[-1] == 1
    if rng is None:
        mock_default_rng.assert_called_once_with()
        assert ifp.rng is mock_rng
    else:
        assert ifp.rng is rng


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


def test_check_fields_logU():
    """Assert an error is raised if logW is missing"""
    add_extra_parameters_to_live_points(["logQ", "logW"])
    with pytest.raises(
        RuntimeError, match=r"logU field missing in non-sampling parameters."
    ):
        IFP._check_fields()
    reset_extra_live_points_parameters()


def test_initialise(ifp):
    output = "test_init"
    flow_config = {"a": 1}
    training_config = {"b": 2}
    ifp.initialised = False
    ifp.output = output
    ifp.flow_config = flow_config
    ifp.training_config = training_config
    flow = MagicMock()
    with (
        patch(
            "nessai.proposal.importance.Proposal.initialise",
        ) as mock_init,
        patch(
            "nessai.proposal.importance.ImportanceFlowModel", return_value=flow
        ) as mock_flow,
    ):
        IFP.initialise(ifp)
    ifp._check_fields.assert_called_once()
    ifp.verify_rescaling.assert_called_once()
    # Make sure flow is initialised
    mock_flow.assert_called_once_with(
        flow_config=flow_config, training_config=training_config, output=output
    )
    flow.initialise.assert_called_once()
    mock_init.assert_called_once()


def test_already_initialised(ifp):
    """Assert functions are not called"""
    ifp.initialised = True

    with (
        patch(
            "nessai.proposal.importance.Proposal.initialise",
        ) as mock_init,
        patch(
            "nessai.proposal.importance.ImportanceFlowModel",
        ) as mock_flow,
    ):
        IFP.initialise(ifp)
    mock_init.assert_not_called()
    mock_flow.assert_not_called()
    ifp._check_fields.assert_called_once()
    ifp.verify_rescaling.assert_not_called()
