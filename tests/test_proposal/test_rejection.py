# -*- coding: utf-8 -*-
"""
Test the rejection proposal class.
"""
import pytest

from nessai.proposal import RejectionProposal


@pytest.fixture()
def poolsize():
    return 10


@pytest.fixture(scope='function')
def proposal(model, poolsize):
    return RejectionProposal(model, poolsize=poolsize)


def test_populate(proposal, poolsize):
    """Test the population process"""
    proposal.populate(N=poolsize)

    assert proposal.samples.size == poolsize
    assert proposal.populated
    assert (proposal.samples['logW'] == 0.).all()


def test_populate_importance(proposal, poolsize):
    """Test the population process for importance sampling"""
    proposal._rejection_sampling = False
    proposal.populate(N=poolsize)
    target_log_w = (
        proposal.model.log_prior(proposal.samples)
        - proposal.log_proposal(proposal.samples)
    )

    assert proposal.samples.size == poolsize
    assert proposal.populated
    assert (proposal.samples['logW'] == target_log_w).all()


def test_draw(model, proposal, poolsize):
    """Test the draw method"""
    old_point = model.new_point()
    assert not proposal.populated
    proposal.draw(old_point)
    assert proposal.populated
    [proposal.draw(old_point) for _ in range(poolsize - 1)]
    assert not proposal.indices
    assert not proposal.populated
