from unittest.mock import patch

import pytest

from nessai.proposal.flowproposal.flowproposal import FlowProposal
from nessai.proposal.flowproposal.truncation import (
    LatentRadiusTruncation,
    TruncationScheme,
)


@pytest.mark.parametrize(
    "expansion_fraction, fuzz", [(None, 2.0), (0.5, 1.5**0.5)]
)
def test_set_rescaling(proposal, expansion_fraction, fuzz):
    proposal.prime_dims = 2
    proposal._truncation_scheme = TruncationScheme(
        [
            LatentRadiusTruncation(
                fuzz=2.0,
                fixed_radius=1.0,
                radius_mode="fixed",
                expansion_fraction=expansion_fraction,
            )
        ]
    )
    with patch(
        "nessai.proposal.flowproposal.flowproposal.BaseFlowProposal.set_rescaling"  # noqa: E501
    ) as mock_parent:
        FlowProposal.set_rescaling(proposal)
    mock_parent.assert_called_once()
    rule = proposal._truncation_scheme.get_rule("latent_radius")
    assert rule.fuzz == fuzz
