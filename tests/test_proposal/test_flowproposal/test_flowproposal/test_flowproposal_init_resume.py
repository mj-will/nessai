# -*- coding: utf-8 -*-
"""Test methods related to initialising and resuming the proposal method"""

from unittest.mock import patch

import numpy as np
import pytest

from nessai.proposal import FlowProposal


def test_init(model):
    """Test init with some kwargs"""
    fp = FlowProposal(model, poolsize=1000)
    assert fp.model == model
    assert fp.poolsize == 1000


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("mask", [None, [1, 0]])
def test_get_state(proposal, populated, mask):
    """Test the get state method used for pickling the proposal.

    Tests cases where the proposal is and isn't populated.
    """
    parent_state = {"a": "val"}
    with patch(
        "nessai.proposal.flowproposal.base.BaseFlowProposal.__getstate__",
        return_value=parent_state,
    ) as mock:
        state = FlowProposal.__getstate__(proposal)

    mock.assert_called_once()
    assert state["_draw_func"] is None
    assert state["_populate_dist"] is None
    assert state["alt_dist"] is None
    assert state["a"] == "val"


@pytest.mark.integration_test
@pytest.mark.parametrize("reparameterisation", [False, True])
@pytest.mark.parametrize("init", [False, True])
@pytest.mark.parametrize(
    "latent_prior", ["truncated_gaussian", "uniform_nball", "gaussian"]
)
def test_resume_pickle(model, tmpdir, reparameterisation, init, latent_prior):
    """Test pickling and resuming the proposal.

    Tests both with and without reparameterisations and before and after
    initialise has been called.
    """
    import pickle

    output = tmpdir.mkdir("test_integration")
    if reparameterisation:
        reparameterisations = {"default": {"parameters": model.names}}
    else:
        reparameterisations = None

    if latent_prior != "truncated_gaussian":
        constant_volume_mode = False
    else:
        constant_volume_mode = True

    proposal = FlowProposal(
        model,
        poolsize=1000,
        plot=False,
        expansion_fraction=1,
        output=output,
        reparameterisations=reparameterisations,
        latent_prior=latent_prior,
        constant_volume_mode=constant_volume_mode,
    )
    if init:
        proposal.initialise()

    proposal.mask = None
    proposal.resume_populated = False

    proposal_data = pickle.dumps(proposal)
    proposal_re = pickle.loads(proposal_data)
    proposal_re.resume(model, {})

    assert proposal._plot_pool == proposal_re._plot_pool
    assert proposal._plot_training == proposal_re._plot_training

    if init:
        assert proposal.fuzz == proposal_re.fuzz
        assert proposal.prime_parameters == proposal_re.rescaled_names


def test_reset(proposal):
    """Test reset method"""
    with patch(
        "nessai.proposal.flowproposal.base.BaseFlowProposal.reset"
    ) as mock:
        FlowProposal.reset(proposal)
    mock.assert_called_once()
    assert proposal.r is np.nan
    assert proposal.alt_dist is None


@pytest.mark.timeout(60)
@pytest.mark.integration_test
@pytest.mark.parametrize(
    "latent_prior", ["truncated_gaussian", "uniform_nball", "gaussian"]
)
def test_reset_integration(tmpdir, model, latent_prior):
    """Test reset method iteration with other methods"""
    flow_config = dict(
        n_neurons=1,
        n_blocks=1,
        n_layers=1,
        batch_norm_between_layers=False,
        linear_transform=None,
    )
    training_config = dict(patience=20)
    output = str(tmpdir.mkdir("reset_integration"))
    poolsize = 2
    drawsize = 100
    if latent_prior != "truncated_gaussian":
        constant_volume_mode = False
    else:
        constant_volume_mode = True
    proposal = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        training_config=training_config,
        plot=False,
        poolsize=poolsize,
        drawsize=drawsize,
        expansion_fraction=None,
        latent_prior=latent_prior,
        constant_volume_mode=constant_volume_mode,
    )

    modified_proposal = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        training_config=training_config,
        plot=False,
        poolsize=poolsize,
        drawsize=drawsize,
        expansion_fraction=None,
        latent_prior=latent_prior,
        constant_volume_mode=constant_volume_mode,
    )
    proposal.initialise()
    modified_proposal.initialise()

    modified_proposal.populate(model.new_point())
    modified_proposal.reset()

    # attributes that should be different
    ignore = [
        "population_time",
        "_reparameterisation",
        "rng",
    ]

    d1 = proposal.__getstate__()
    d2 = modified_proposal.__getstate__()

    for key in ignore:
        d1.pop(key)
        d2.pop(key)

    assert d1 == d2
