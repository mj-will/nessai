# -*- coding: utf-8 -*-
"""Test methods related to initialising and resuming the proposal method"""
import os
import pytest
from unittest.mock import patch, MagicMock

from nessai.proposal import FlowProposal


@pytest.mark.parametrize('kwargs',
                         [{'prior': 'uniform'},
                          {'draw_latent_kwargs': {'var': 4}}
                          ])
def test_init(model, kwargs):
    """Test init with some kwargs"""
    fp = FlowProposal(model, poolsize=1000, **kwargs)
    assert fp.model == model
    # Make sure the dummy kwargs is ignored and not added
    assert getattr(fp, 'prior', None) is None


@pytest.mark.parametrize('ef, fuzz', [(2.0, 3.0**0.5), (False, 2.0)])
@patch('nessai.flowmodel.FlowModel', new=MagicMock())
def test_initialise(tmpdir, proposal, ef, fuzz):
    """Test the initialise method"""
    p = tmpdir.mkdir('test')
    proposal.output = f'{p}/output/'
    proposal.rescaled_dims = 2
    proposal.expansion_fraction = ef
    proposal.fuzz = 2.0
    proposal.flow_config = {'model_config': {}}
    proposal.set_rescaling = MagicMock()
    proposal.verify_rescaling = MagicMock()

    FlowProposal.initialise(proposal)

    proposal.set_rescaling.assert_called_once()
    proposal.verify_rescaling.assert_called_once()
    assert proposal.populated is False
    assert proposal.initialised
    assert proposal.fuzz == fuzz
    assert os.path.exists(f'{p}/output')


def test_resume(proposal):
    """Test the resume method."""
    proposal.initialise = MagicMock()
    proposal.mask = None
    proposal.update_bounds = False
    proposal.weights_file = None
    FlowProposal.resume(proposal, None, {})
    proposal.initialise.assert_called_once()


@pytest.mark.integration_test
def test_resume_integration(model, tmpdir):
    import pickle
    output = tmpdir.mkdir('test_integration')
    proposal = FlowProposal(model, poolsize=1000, plot=False,
                            expansion_fraction=1, output=output)
    proposal.initialise()
    proposal.mask = None
    proposal.resume_populated = False

    proposal_data = pickle.dumps(proposal)
    proposal_re = pickle.loads(proposal_data)
    proposal_re.resume(model, {})

    for attr in ['fuzz', '_plot_pool', '_plot_training', 'rescaled_names']:
        assert getattr(proposal, attr) == getattr(proposal_re, attr)
