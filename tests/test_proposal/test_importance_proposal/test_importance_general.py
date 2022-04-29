# -*- coding: utf-8 -*-
"""
General tests for the importance proposal.
"""

import pytest
from unittest.mock import MagicMock, patch

from nessai.proposal import ImportanceFlowProposal as IFP


def test_init(proposal, model, tmp_path):
    """Assert the correct variables are set and methods called"""
    output = tmp_path / 'test'
    output.mkdir()
    IFP.__init__(proposal, model, output, 100)


def test_check_fields():
    """Assert an error is not raised if logQ or logW are present"""
    fields = ['logL', 'logQ', 'logW']
    with patch("nessai.config.NON_SAMPLING_PARAMETERS", fields):
        IFP._check_fields()


@pytest.mark.parametrize("fields", [['logL', 'logW'], ['logL', 'logQ']])
def test_check_fields_error(fields):
    """Assert errors are raised if logQ or logW are missing"""
    with patch("nessai.config.NON_SAMPLING_PARAMETERS", fields), \
         pytest.raises(RuntimeError) as excinfo:
        IFP._check_fields()
    assert "missing in non-sampling parameters" in str(excinfo.value)


def test_initialise(proposal):
    """Assert initialise calls the correct methods"""

    output = "./"
    config = {"test": 2}

    proposal.initialised = False
    proposal.output = output
    proposal.flow_config = config
    proposal._check_fields = MagicMock()
    proposal.verify_rescaling = MagicMock()

    flow = MagicMock()
    flow.initialise = MagicMock()

    with patch(
        "nessai.proposal.importance.CombinedFlowModel", return_value=flow
    ) as cfm, patch("nessai.proposal.importance.Proposal.initialise") as pinit:
        IFP.initialise(proposal)

    assert proposal.flow is flow

    cfm.assert_called_once_with(config=config, output=output)
    pinit.assert_called_once()
    flow.initialise.assert_called_once()
    proposal._check_fields.assert_called_once()
    proposal.verify_rescaling.assert_called_once()
