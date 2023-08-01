# -*- coding: utf-8 -*-
"""
Integration tests for adding the default reparameterisations
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec

from nessai.model import Model
from nessai.proposal import FlowProposal
from nessai.reparameterisations import default_reparameterisations

# General reparameterisations that do not need extra parameters
general_reparameterisations = {
    k: v
    for k, v in default_reparameterisations.items()
    if k not in ["scale", "rescale", "angle-pair", "scaleandshift"]
}


@pytest.fixture(params=general_reparameterisations.keys())
def reparameterisation(request):
    return request.param


@pytest.fixture
def model():
    m = create_autospec(Model)
    m.names = ["x"]
    m.bounds = {"x": [-1, 1]}
    m.reparameterisations = None
    return m


@pytest.mark.integration_test
def test_configure_reparameterisations(tmpdir, model, reparameterisation):
    """Test adding one of the default reparameterisations.

    Only tests reparameterisations that don't need extra parameters.
    """
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir("test")),
        poolsize=10,
        reparameterisations={"x": reparameterisation},
    )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
@pytest.mark.parametrize("reparameterisation", ["scale", "rescale"])
def test_configure_reparameterisation_scale(tmpdir, model, reparameterisation):
    """Test adding the `Rescale` reparameterisation"""
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir("test")),
        poolsize=10,
        reparameterisations={
            "x": {"reparameterisation": reparameterisation, "scale": 2.0}
        },
    )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
def test_configure_reparameterisation_angle_pair(tmpdir, model):
    """Test adding the `AnglePair` reparameterisation"""
    model.names.append("y")
    model.bounds = {"x": [0, 2 * np.pi], "y": [-np.pi / 2, np.pi / 2]}
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir("test")),
        poolsize=10,
        reparameterisations={
            "x": {"reparameterisation": "angle-pair", "parameters": ["y"]}
        },
    )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
def test_default_reparameterisations(caplog, tmpdir):
    """Assert that by default the reparameterisation is RescaleToBounds"""
    caplog.set_level("INFO")
    model = MagicMock()
    model.names = ["x1", "x10", "x11"]
    model.bounds = {p: [-1, 1] for p in model.names}
    model.reparameterisations = None
    proposal = FlowProposal(
        model, poolsize=100, output=str(tmpdir.mkdir("test"))
    )
    # Mocked model so can't verify rescaling
    proposal.verify_rescaling = MagicMock()
    proposal.initialise()
    reparams = list(proposal._reparameterisation.values())
    assert len(reparams) == 1
    assert reparams[0].parameters == ["x1", "x10", "x11"]
    assert proposal.rescale_parameters == ["x1", "x10", "x11"]
