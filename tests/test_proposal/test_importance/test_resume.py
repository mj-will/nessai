"""Test the properties defined in ImportanceFlowProposal"""

import pickle
from unittest.mock import MagicMock, patch

from nessai.proposal.importance import ImportanceFlowProposal as IFP
import pytest


def test_resume(ifp, model, tmp_path):

    flow_config = dict(patience=10, model_config=dict(n_inputs=2))

    ifp.initialise = MagicMock()
    ifp.flow = MagicMock()
    ifp.flow.resume = MagicMock()

    path = tmp_path / "test_resume"

    with patch("nessai.proposal.importance.Proposal.resume") as mock_parent:
        IFP.resume(ifp, model, flow_config, weights_path=path)

    mock_parent.assert_called_once_with(model)
    ifp.initialise.assert_called_once_with()

    assert ifp.flow_config is flow_config
    ifp.flow.resume.assert_called_once_with(
        flow_config["model_config"],
        weights_path=path,
    )


@pytest.mark.integration_test
@pytest.mark.usefixtures("ins_parameters")
def test_getstate_integration(tmp_path, model):

    ifp = IFP(
        model,
        tmp_path / "test_resume",
        100,
        weighted_kl=False,
    )
    ifp.initialise()

    for _ in range(4):
        ifp.train(model.new_point(10), max_epochs=2)
        ifp.draw(10)

    out = pickle.dumps(ifp)

    ifp_resume = pickle.loads(out)
    ifp_resume.resume(model, {})

    assert len(ifp_resume.flow.models) == 4
