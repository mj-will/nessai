from unittest.mock import patch

import pytest

from nessai.proposal.flowproposal.flowproposal import FlowProposal


@pytest.mark.parametrize(
    "expansion_fraction, fuzz", [(None, 2.0), (0.5, 1.5**0.5)]
)
def test_set_rescaling(proposal, expansion_fraction, fuzz):
    proposal.rescaled_dims = 2
    proposal.expansion_fraction = expansion_fraction
    proposal.fuzz = 2.0
    with patch(
        "nessai.proposal.flowproposal.flowproposal.BaseFlowProposal.set_rescaling"  # noqa: E501
    ) as mock_parent:
        FlowProposal.set_rescaling(proposal)
    mock_parent.assert_called_once()
    proposal.configure_constant_volume.assert_called_once()
    assert proposal.fuzz == fuzz
