"""Test methods related to initialising and resuming the proposal method"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai.proposal.flowproposal.base import BaseFlowProposal


@pytest.mark.parametrize(
    "value, expected", [(True, True), (False, False), (None, False)]
)
def test_init_use_default_reparams(model, proposal, value, expected):
    """Assert use_default_reparameterisations is set correctly"""
    proposal.use_default_reparameterisations = False
    BaseFlowProposal.__init__(
        proposal, model, poolsize=10, use_default_reparameterisations=value
    )
    assert proposal.use_default_reparameterisations is expected


def test_initialise(tmpdir, proposal):
    """Test the initialise method"""
    p = tmpdir.mkdir("test")
    proposal.initialised = False
    proposal.output = os.path.join(p, "output")
    proposal.flow_config = {}
    proposal.training_config = {}
    proposal.set_rescaling = MagicMock()
    proposal.verify_rescaling = MagicMock()
    proposal.update_flow_config = MagicMock()
    fm = MagicMock()
    fm.initialise = MagicMock()
    proposal._FlowModelClass = MagicMock(new=fm)

    BaseFlowProposal.initialise(proposal, resumed=False)

    proposal.set_rescaling.assert_called_once()
    proposal.verify_rescaling.assert_called_once()
    proposal.update_flow_config.assert_called_once()
    proposal._FlowModelClass.assert_called_once_with(
        flow_config=proposal.flow_config,
        training_config=proposal.training_config,
        output=proposal.output,
        rng=proposal.rng,
    )
    proposal.flow.initialise.assert_called_once()
    assert proposal.populated is False
    assert proposal.initialised
    assert os.path.exists(os.path.join(p, "output"))


def test_resume(proposal):
    """Test the resume method."""
    proposal.initialise = MagicMock()
    proposal.mask = [1, 0]
    proposal.update_bounds = False
    proposal.weights_file = None
    model = MagicMock()
    with patch("nessai.proposal.base.Proposal.resume") as mock:
        BaseFlowProposal.resume(proposal, model, {})
    mock.assert_called_once_with(model)
    proposal.initialise.assert_called_once()
    assert np.array_equal(proposal.flow_config["mask"], np.array([1, 0]))


@patch("os.path.exists", return_value=True)
def test_resume_w_weights(osexist, proposal):
    """Test the resume method with weights"""
    proposal.initialise = MagicMock()
    proposal.flow = MagicMock()
    proposal.mask = None
    proposal.update_bounds = False
    proposal.weights_file = None
    model = MagicMock()
    with patch("nessai.proposal.base.Proposal.resume") as mock:
        BaseFlowProposal.resume(proposal, model, {}, weights_file="weights.pt")
    mock.assert_called_once_with(model)
    osexist.assert_called_once_with("weights.pt")
    proposal.initialise.assert_called_once()
    proposal.flow.reload_weights.assert_called_once_with("weights.pt")


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("mask", [None, [1, 0]])
def test_get_state(proposal, populated, mask):
    """Test the get state method used for pickling the proposal.

    Tests cases where the proposal is and isn't populated.
    """

    proposal.populated = populated
    proposal.indices = [1, 2]
    proposal._reparameterisation = MagicMock()
    proposal.model = MagicMock()
    proposal._flow_config = {}
    proposal.initialised = True
    proposal.flow = MagicMock()
    proposal.flow.weights_file = "file"

    if mask is not None:
        proposal.flow.flow_config = {"mask": mask}

    state = BaseFlowProposal.__getstate__(proposal)

    assert state["resume_populated"] is populated
    assert state["initialised"] is False
    assert state["weights_file"] == "file"
    assert state["mask"] is mask
    assert "model" not in state
    assert "flow" not in state
    assert "_flow_config" not in state


def test_reset(proposal):
    """Test reset method"""
    proposal.x = 1
    proposal.samples = 2
    proposal.populated = True
    proposal.populated_count = 10
    proposal._reparameterisation = MagicMock()
    BaseFlowProposal.reset(proposal)
    assert proposal.x is None
    assert proposal.samples is None
    assert proposal.populated is False
    assert proposal.populated_count == 0
    assert proposal._checked_population
    proposal._reparameterisation.reset.assert_called_once()


def test_populate_error(proposal):
    with pytest.raises(NotImplementedError):
        BaseFlowProposal.populate(proposal, 1.0, n_samples=10)
