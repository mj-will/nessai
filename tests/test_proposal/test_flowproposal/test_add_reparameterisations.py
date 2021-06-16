# -*- coding: utf-8 -*-
"""
Integration tests for adding the default reparameterisations
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.model import Model
from nessai.proposal import FlowProposal
from nessai.reparameterisations import default_reparameterisations

# General reparameterisations that do not need extra parameters
general_reparameterisations = \
    {k: v for k, v in default_reparameterisations.items()
        if k not in ['scale', 'rescale', 'angle-pair']}


@pytest.fixture(params=general_reparameterisations.keys())
def reparameterisation(request):
    return request.param


@pytest.fixture
def model():
    m = create_autospec(Model)
    m.names = ['x']
    m.bounds = {'x': [-1, 1]}
    m.reparameterisations = None
    return m


@pytest.mark.integration_test
def test_configure_reparameterisations(tmpdir, model, reparameterisation):
    """Test adding one of the default reparameterisations.

    Only tests reparameterisations that don't need extra parameters.
    """
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir('test')),
        poolsize=10,
        reparameterisations={'x': reparameterisation}
        )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
@pytest.mark.parametrize('reparameterisation', ['scale', 'rescale'])
def test_configure_reparameterisation_scale(tmpdir, model, reparameterisation):
    """Test adding the `Rescale` reparameterisation"""
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir('test')),
        poolsize=10,
        reparameterisations={
            'x': {'reparameterisation': reparameterisation, 'scale': 2.0}
        }
        )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
def test_configure_reparameterisation_angle_pair(tmpdir, model):
    """Test adding the `AnglePair` reparameterisation"""
    model.names.append('y')
    model.bounds = {'x': [0, 2 * np.pi], 'y': [-np.pi / 2, np.pi / 2]}
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir('test')),
        poolsize=10,
        reparameterisations={
            'x': {'reparameterisation': 'angle-pair', 'parameters': ['y']}
        }
        )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None
