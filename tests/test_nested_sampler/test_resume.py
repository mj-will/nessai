# -*- coding: utf-8 -*-
"""
Tests related to resuming.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from nessai.nestedsampler import NestedSampler


@pytest.fixture
def complete_sampler(model, tmpdir):
    """Complete instance of NestedSampler"""
    output = tmpdir.mkdir('output')
    ns = NestedSampler(model, output=output, poolsize=10)
    ns.initialise()
    return ns


def test_check_resume(sampler):
    """Test check resume method"""
    sampler.uninformed_sampling = False
    sampler.check_proposal_switch = MagicMock()
    sampler.resumed = True
    sampler._flow_proposal = MagicMock()
    sampler._flow_proposal.populated = False
    sampler._flow_proposal._resume_populated = True
    sampler._flow_proposal.indices = [1, 2, 3]

    NestedSampler.check_resume(sampler)

    sampler.check_proposal_switch.assert_called_once_with(force=True)
    assert sampler.resumed is False
    assert sampler._flow_proposal.populated is True


def test_check_resume_no_indices(sampler):
    """Test check resume method"""
    sampler.uninformed_sampling = True
    sampler.resumed = True
    sampler._flow_proposal = MagicMock()
    sampler._flow_proposal.populated = False
    sampler._flow_proposal._resume_populated = True
    sampler._flow_proposal.indices = []

    NestedSampler.check_resume(sampler)

    assert sampler.resumed is False
    assert sampler._flow_proposal.populated is False


def test_resume(model):
    """Test the resume method"""
    obj = MagicMock()
    obj._uninformed_proposal = MagicMock()
    obj._uninformed_proposal.resume = MagicMock()
    obj._flow_proposal = MagicMock()
    obj._flow_proposal.resume = MagicMock()

    model = MagicMock()

    with patch(
        'nessai.nestedsampler.BaseNestedSampler.resume', return_value=obj
    ) as mock_resume:
        out = NestedSampler.resume('test.pkl', model)

    mock_resume.assert_called_once_with('test.pkl', model)

    out._flow_proposal.resume.assert_called_once_with(model, {}, None)
    out._uninformed_proposal.resume.assert_called_once_with(model)
    assert out.resumed is True


def test_get_state(sampler):
    """Test the getstate method used for pickling.

    It should remove the model.
    """
    sampler.model = MagicMock()
    state = NestedSampler.__getstate__(sampler)
    assert 'model' not in state


@pytest.mark.integration_test
def test_checkpoint_integration(complete_sampler):
    """Integration test for checkpointing the sampler."""
    complete_sampler.checkpoint()
    resume_file = os.path.join(
        complete_sampler.output, complete_sampler.resume_file)
    assert os.path.exists(resume_file)


@pytest.mark.integration_test
def test_checkpoint_resume_integration(complete_sampler, model):
    """Integration test for checkpointing the sampler."""
    complete_sampler.likelihood_evaluations = [1, 2]
    complete_sampler.checkpoint()
    resume_file = os.path.join(
        complete_sampler.output, complete_sampler.resume_file)
    assert os.path.exists(resume_file)
    ns = NestedSampler.resume(resume_file, model)
    assert ns is not None
